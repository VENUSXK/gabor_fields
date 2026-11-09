#!/usr/bin/env python3
"""Unified local/global SVD presentation exports, drawn directly with Pillow."""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from local_svd_core import analyze_local, _device
from svd_reconstruction import reconstruct_slices, block_modes


def font(size=20):
    for name in ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def load_stack(directory, pattern):
    files = sorted(Path(directory).glob(pattern), key=lambda p: [
        int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", p.name)])
    if not files:
        raise ValueError(f"No input images in {directory} matching {pattern}")
    images = []
    for path in files:
        with Image.open(path) as image:
            images.append(np.asarray(image.convert("L"), dtype=np.float32) / 255)
    if len({a.shape for a in images}) != 1:
        raise ValueError("All input slices must have the same dimensions")
    return np.stack(images), files


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def psnr_from_mse(mse):
    return -10 * math.log10(mse) if mse > 0 else float("inf")


def gray(values):
    return Image.fromarray(np.uint8(np.rint(np.clip(values, 0, 1) * 255)))


@lru_cache(maxsize=1)
def error_palette():
    points = np.array([[0, 0, 0], [52, 15, 95], [145, 38, 103],
                       [230, 91, 55], [253, 184, 55], [252, 255, 180]], float)
    v = np.linspace(0,len(points)-1,256)
    index = np.minimum(v.astype(int), len(points)-2)
    rgb = points[index] * (1-(v-index)[..., None]) + points[index+1] * (v-index)[..., None]
    return np.uint8(rgb).ravel().tolist()


def heat(values, vmax):
    """Fixed indexed PNG palette; avoids large RGB floating-point temporaries."""
    image=Image.fromarray(np.uint8(np.rint(np.clip(values/vmax,0,1)*255)))
    image.putpalette(error_palette())
    return image


def signed(values):
    scale = max(float(np.max(abs(values))), 1e-30)
    t = np.clip(values / scale, -1, 1)[..., None]
    white = np.full((*values.shape, 3), 245.0)
    color = np.where(t >= 0, np.array([210, 35, 45]), np.array([35, 75, 205]))
    return Image.fromarray(np.uint8(white * (1-abs(t)) + color * abs(t)))


def save_png(image, path, description=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Description", description)
    image.save(path, compress_level=1, pnginfo=metadata)


def sheet(images, labels, title, path, columns=3, cell_width=440, nearest=False):
    rows = math.ceil(len(images) / columns)
    cell_height = 360
    canvas = Image.new("RGB", (columns * cell_width, 85 + rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), title, fill="black", font=font(20))
    for index, (image, label) in enumerate(zip(images, labels)):
        x = (index % columns) * cell_width
        y = 75 + (index // columns) * cell_height
        tile = image.convert("RGB")
        if nearest:
            factor = min((cell_width-20)/tile.width, (cell_height-65)/tile.height)
            tile = tile.resize((max(1,round(tile.width*factor)),max(1,round(tile.height*factor))),Image.Resampling.NEAREST)
        else:
            tile.thumbnail((cell_width-20, cell_height-65), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x + (cell_width-tile.width)//2, y+48))
        draw.multiline_text((x+10, y), label, fill="black", font=font(15), spacing=3)
    save_png(canvas, path, title)


def pixel_error_legend(vmax, path):
    ramp = np.repeat(np.linspace(0, vmax, 700)[None, :], 35, 0)
    out = Image.new("RGB", (760, 125), "white")
    out.paste(heat(ramp, vmax), (30, 45))
    draw = ImageDraw.Draw(out)
    draw.text((30, 10), "Absolute voxel error; same scale for all configurations", fill="black", font=font(18))
    draw.text((30, 88), "0", fill="black", font=font(17))
    draw.text((550, 88), f">= {vmax:g} (saturated)", fill="black", font=font(17))
    save_png(out, path)


def table_image(rows, columns, title, path):
    widths = [max(120, min(310, len(label)*11+25)) for _, label in columns]
    out = Image.new("RGB", (sum(widths)+40, 100+35*(len(rows)+1)), "white")
    draw = ImageDraw.Draw(out)
    draw.text((15, 14), title, fill="black", font=font(20))
    for j, row in enumerate([dict(columns)] + rows):
        x = 20
        y = 65+j*35
        draw.rectangle((20, y, out.width-20, y+34), fill="#dde7f1" if j == 0 else ("#f1f4f7" if j%2 else "white"))
        for (key, label), width in zip(columns, widths):
            value = label if j == 0 else row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.3f}" if math.isfinite(value) else "exact"
            draw.text((x+5, y+7), str(value), fill="black", font=font(16))
            x += width
    save_png(out, path, title)


def quality_chart(rows, title, path):
    """Small report-ready PSNR/rank chart, drawn with Pillow primitives."""
    valid=[r for r in rows if math.isfinite(r["psnr_db"]) and r["requested_rank"]>0]
    canvas=Image.new("RGB",(1200,720),"white")
    draw=ImageDraw.Draw(canvas)
    draw.text((30,15),title,fill="black",font=font(24))
    if not valid:
        save_png(canvas,path)
        return
    xmax=max(r["requested_rank"] for r in valid)
    ymin=5*math.floor(min(r["psnr_db"] for r in valid)/5)
    ymax=max(ymin+5,5*math.ceil(max(r["psnr_db"] for r in valid)/5))
    for y in np.linspace(ymin,ymax,6):
        py=590-(y-ymin)/(ymax-ymin)*490
        draw.line((90,py,910,py),fill="#dddddd")
        draw.text((25,py-10),f"{y:.1f}",fill="black",font=font(17))
    for rank in sorted({r["requested_rank"] for r in valid}):
        x=90+rank/xmax*820
        draw.text((x-8,603),str(rank),fill="black",font=font(17))
    draw.line((90,90,90,590,920,590),fill="black",width=2)
    draw.text((400,647),"Retained rank (mean excluded for centered)",fill="black",font=font(19))
    draw.text((22,65),"PSNR dB",fill="black",font=font(18))
    colors=["#1f77b4","#d62728","#2ca02c","#9467bd","#ff7f0e","#8c564b"]
    names=list(dict.fromkeys(r["blocksize"] for r in valid))
    for i,name in enumerate(names):
        values=sorted([r for r in valid if r["blocksize"]==name],key=lambda r:r["requested_rank"])
        points=[(90+r["requested_rank"]/xmax*820,590-(r["psnr_db"]-ymin)/(ymax-ymin)*490) for r in values]
        color=colors[i%len(colors)]
        if len(points)>1:
            draw.line(points,fill=color,width=3)
        for x,y in points:
            draw.ellipse((x-5,y-5,x+5,y+5),fill=color)
        draw.text((945,110+i*38),f"block {name}",fill=color,font=font(19))
    draw.text((90,682),"All original voxels; infinite/full reconstruction points omitted.",fill="#555555",font=font(16))
    save_png(canvas,path,title)


def summarize(result, analysis, rank, slice_sse, clip_count, name):
    data = result["analyses"][analysis]
    bounds = data["bounds"]
    sizes = bounds[:, 3:] - bounds[:, :3]
    pixels = sizes[:, 1]*sizes[:, 2]
    actual = np.minimum(rank, data["max_rank"])
    params = int(np.sum(actual * (pixels + sizes[:, 0] + 1)))
    if analysis == "axial_centered":
        params += int(pixels.sum())
    voxels = int(np.prod(result["metadata"]["shape"]))
    mse = float(slice_sse.sum()) / voxels
    return dict(configuration=name, kind="global_xy" if name.startswith("global") else "local",
                blocksize="full_image" if name.startswith("global") else result["metadata"]["resolution"],
                slab=result["metadata"]["slab"], analysis=analysis, requested_rank=rank,
                actual_rank_min=int(actual.min()), actual_rank_max=int(actual.max()),
                block_count=len(bounds), voxel_count=voxels,
                psnr_db=psnr_from_mse(mse), rmse=math.sqrt(mse), sse=float(slice_sse.sum()),
                factor_scalar_count=params, scalar_compression_ratio=voxels/params if params else float("inf"),
                estimated_factor_bytes_float32=params*4,
                uint8_source_to_float32_factor_ratio=voxels/(params*4) if params else float("inf"),
                full_rank_block_fraction=float(np.mean(actual >= data["max_rank"])),
                image_outside_unit_interval_count=int(clip_count),
                metric_domain="all original voxels; unclipped floating-point reconstruction",
                evaluation="in_sample_reconstruction_not_held_out_cross_validation")


def export_templates(stack, result, analysis, directory, config_name, count, modes):
    data = result["analyses"][analysis]
    indices = np.flatnonzero(data["full_block"] & ~data["near_zero"])
    if not indices.size:
        indices = np.arange(len(data["bounds"]))
    j = int(np.argmin(abs(np.asarray(result["metadata"]["ranks"])-2)))
    errors = np.sqrt(data["sse"][:, j]/data["voxel_count"])
    ordered = indices[np.argsort(errors[indices])]
    quantiles = [0.5, 0.9, 1.0][:count]
    selected = list(dict.fromkeys(int(ordered[round(q*(len(ordered)-1))]) for q in quantiles))
    gallery, labels, manifest = [], [], []
    for index in selected:
        b = data["bounds"][index]
        z0, y0, x0, z1, y1, x1 = map(int, b)
        prefix = f"{config_name}__{analysis}__block{index:06d}__x{x0:04d}-{x1:04d}__y{y0:04d}-{y1:04d}__z{z0:03d}-{z1:03d}"
        mode_data = block_modes(stack, b, analysis, modes)
        perblock, perlabels = [], []
        if analysis == "axial_centered":
            image = gray(mode_data["mean"])
            save_png(image, directory / f"{prefix}__mean_template_native.png")
            perblock.append(image)
            perlabels.append("Mean template (not included in rank)")
        for k, spatial in enumerate(mode_data["spatial"]):
            image = signed(spatial)
            mode_name = f"{prefix}__mode{k+1:02d}"
            save_png(image, directory / f"{mode_name}__spatial_signed_native.png",
                     "Signed spatial singular vector; independent symmetric normalization. White=zero, red=positive, blue=negative.")
            save_png(image.resize((256, 256), Image.Resampling.NEAREST),
                     directory / f"{mode_name}__spatial_signed_preview256.png")
            perblock.append(image)
            perlabels.append(f"Mode {k+1}; sigma={mode_data['sigma'][k]:.5g}")
            gallery.append(image)
            labels.append(f"block {index}, mode {k+1}\nxy=({x0},{y0}), z=[{z0},{z1})")
        coefficients = []
        for zi in range(z1-z0):
            row = {"slice_z": z0+zi}
            for k in range(len(mode_data["spatial"])):
                row[f"v_mode_{k+1}"] = float(mode_data["axial"][k, zi])
                row[f"sigma_v_mode_{k+1}"] = float(mode_data["sigma"][k]*mode_data["axial"][k, zi])
            coefficients.append(row)
        write_csv(directory / f"{prefix}__axial_coefficients.csv", coefficients)
        # Draw coefficient curves directly with Pillow.
        profiles = Image.new("RGB", (1000, 430), "white")
        draw = ImageDraw.Draw(profiles)
        draw.text((20, 12), f"{config_name} / {analysis} / block {index}: axial v(z)", fill="black", font=font(19))
        axis = mode_data["axial"]
        scale = max(float(np.max(abs(axis))) if axis.size else 0, 1e-12)
        draw.line((65, 65, 65, 355, 950, 355), fill="black", width=2)
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
        for k, v in enumerate(axis):
            points = [(65 + zi*885/max(len(v)-1, 1), 210-float(value)*140/scale) for zi, value in enumerate(v)]
            if len(points)>1:
                draw.line(points, fill=colors[k%len(colors)], width=3)
            for px, py in points:
                draw.ellipse((px-4,py-4,px+4,py+4), fill=colors[k%len(colors)])
            draw.text((80+k*145, 385), f"mode {k+1}", fill=colors[k%len(colors)], font=font(18))
        draw.text((5, 65), f"+{scale:.2f}", fill="black", font=font(14))
        draw.text((5, 205), "0", fill="black", font=font(14))
        draw.text((5, 340), f"-{scale:.2f}", fill="black", font=font(14))
        for zi in range(z1-z0):
            if zi % max(1, (z1-z0)//8)==0 or zi==z1-z0-1:
                draw.text((60+zi*885/max(z1-z0-1,1), 360), str(z0+zi), fill="black", font=font(13))
        save_png(profiles, directory / f"{prefix}__axial_profiles.png")
        sheet(perblock, perlabels, f"{prefix}\nSigned templates: red positive / blue negative; independently normalized",
              directory / f"{prefix}__templates_contact.png", columns=min(4,len(perblock)), nearest=True)
        manifest.append(dict(block_index=index, x0=x0,x1=x1,y0=y0,y1=y1,z0=z0,z1=z1,
                             selection="median/p90/maximum rank-2 RMSE among full blocks", reference_rmse=float(errors[index])))
    if gallery:
        sheet(gallery, labels, f"{config_name} / {analysis}: selected local spatial templates",
              directory / f"{config_name}__{analysis}__template_gallery.png", columns=max(1,modes), nearest=True)
    write_csv(directory / "selected_blocks.csv", manifest)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent/"presentation_results")
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--resolution", nargs="+", type=int, default=[128,64,32,16])
    parser.add_argument("--slab", nargs="+", type=int, default=[4,8,16,30])
    parser.add_argument("--ranks", nargs="+", type=int, default=[1,2,4,8,12,16])
    parser.add_argument("--slices", nargs="+", default=["all"], help="all, or zero-based slice indices")
    parser.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--template-blocks", type=int, choices=[1,2,3], default=3)
    parser.add_argument("--template-modes", type=int, default=4)
    parser.add_argument("--error-vmax", type=float, default=0.1)
    parser.add_argument("--no-global", action="store_true")
    args = parser.parse_args(argv)
    if any(v<=0 for v in args.resolution+args.slab) or any(v<0 for v in args.ranks):
        parser.error("resolution/slab must be positive and ranks nonnegative")
    if args.batch_size<=0 or args.template_modes<=0 or not math.isfinite(args.error_vmax) or args.error_vmax<=0:
        parser.error("batch-size, template-modes and error-vmax must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    stack, files = load_stack(args.input, args.pattern)
    depth,height,width = stack.shape
    slices = list(range(depth)) if args.slices==["all"] else sorted(set(map(int,args.slices)))
    if not slices or any(z<0 or z>=depth for z in slices):
        raise ValueError("Invalid slice index")
    device = str(_device(args.device))
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(shape=list(stack.shape), selected_slices=slices, actual_device=device,
                  input_files=[str(p.resolve()) for p in files],
                  renderer="Pillow only; no Matplotlib", evaluation="in-sample parameter comparison, not holdout CV",
                  status="running")
    (root/"manifest.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    pixel_error_legend(args.error_vmax, root/"00_error_color_scale.png")
    for z in slices:
        save_png(gray(stack[z]), root/"originals"/f"original__z{z:03d}.png")
    configs = [(f"block{r:03d}x{r:03d}__slab{s:03d}",r,s) for r in dict.fromkeys(args.resolution)
               for s in dict.fromkeys(args.slab)]
    if not args.no_global:
        configs += [(f"globalXY{width}x{height}__slab{s:03d}",max(height,width),s)
                    for s in dict.fromkeys(args.slab)]
    all_rows, per_slice = [], []
    for ci,(name,res,slab) in enumerate(configs,1):
        print(f"[{ci}/{len(configs)}] {name} ({device})",flush=True)
        result = analyze_local(stack,res,slab,ranks=args.ranks,device=device,batch_size=args.batch_size)
        config_dir = root/"configurations"/name
        for analysis in ("raw","axial_centered"):
            ranks = sorted({r for r in args.ranks if r<=min(slab,depth)} | ({0} if analysis=="axial_centered" else set()))
            if not ranks:
                continue
            recon = reconstruct_slices(stack,result,analysis,ranks,slices,device,args.batch_size)
            analysis_dir = config_dir/analysis
            export_templates(stack,result,analysis,analysis_dir/"templates",name,args.template_blocks,args.template_modes)
            rows = []
            rank_contact, contact_labels = [], []
            for ri,rank in enumerate(ranks):
                row = summarize(result,analysis,rank,recon["slice_sse"][ri],recon["outside_unit_interval"][ri],name)
                all_rows.append(row)
                rows.append(row)
                prefix = f"{name}__{analysis}__rank{rank:03d}"
                folder = analysis_dir/f"rank{rank:03d}"
                for z in range(depth):
                    mse = float(recon["slice_sse"][ri,z])/(height*width)
                    per_slice.append(dict(configuration=name,blocksize=row["blocksize"],slab=slab,
                                          analysis=analysis,rank=rank,z=z,psnr_db=psnr_from_mse(mse),rmse=math.sqrt(mse)))
                for si,z in enumerate(slices):
                    reconstructed = recon["images"][ri,si]
                    error = abs(stack[z]-reconstructed)
                    reconstructed_image = gray(reconstructed)
                    error_image = heat(error,args.error_vmax)
                    base = f"{prefix}__z{z:03d}"
                    save_png(reconstructed_image,folder/f"{base}__reconstruction_native.png",
                             "Clipped/rounded 8-bit preview; CSV metrics use unclipped floating-point voxels.")
                    save_png(error_image,folder/f"{base}__absolute_error_fixedscale_native.png",
                             f"Absolute unquantized error, shared range [0,{args.error_vmax}], saturated above vmax.")
                # One row per selected key slice: original / reconstruction / fixed-scale error.
                representative = sorted(set([slices[0],slices[len(slices)//2],slices[-1]]))
                panel_images,panel_labels = [],[]
                for z in representative:
                    si=slices.index(z)
                    reconstructed=recon["images"][ri,si]
                    p=psnr_from_mse(float(recon["slice_sse"][ri,z])/(height*width))
                    panel_images += [gray(stack[z]),gray(reconstructed),heat(abs(stack[z]-reconstructed),args.error_vmax)]
                    panel_labels += [f"Original z={z}",f"Reconstruction z={z}; PSNR={p:.3f} dB",f"Absolute error z={z}; vmax={args.error_vmax}"]
                sheet(panel_images,panel_labels,f"{prefix}\nVolume PSNR={row['psnr_db']:.3f} dB; scalar ratio={row['scalar_compression_ratio']:.3f}x; mean stored for centered",
                      folder/f"{prefix}__original_reconstruction_error_contact.png")
                mid=len(slices)//2
                rank_contact.append(gray(recon["images"][ri,mid]))
                contact_labels.append(f"rank={rank}, z={slices[mid]}\nVolume PSNR={row['psnr_db']:.3f} dB")
            sheet(rank_contact,contact_labels,f"{name} / {analysis}: rank comparison",
                  analysis_dir/f"{name}__{analysis}__ranks_reconstruction_contact.png",columns=min(4,len(ranks)))
            write_csv(analysis_dir/f"{name}__{analysis}__rank_comparison.csv",rows)
            table_image(rows,[("requested_rank","Rank"),("psnr_db","PSNR dB"),("scalar_compression_ratio","Scalar ratio"),
                              ("uint8_source_to_float32_factor_ratio","uint8/fp32 bytes"),("full_rank_block_fraction","Full-rank fraction")],
                        f"{name} / {analysis}: in-sample reconstruction",analysis_dir/f"{name}__{analysis}__metrics_table.png")
            del reconstructed, recon
        write_csv(root/"cross_comparison.csv",all_rows)
        write_csv(root/"per_slice_metrics.csv",per_slice)
        print(f"  exported {name}",flush=True)
    for analysis in ("raw","axial_centered"):
        for slab in dict.fromkeys(args.slab):
            subset=[r for r in all_rows if r["analysis"]==analysis and r["slab"]==slab]
            table_image(subset,[("blocksize","Blocksize"),("requested_rank","Rank"),("psnr_db","PSNR dB"),
                                ("scalar_compression_ratio","Scalar ratio"),("uint8_source_to_float32_factor_ratio","uint8/fp32 bytes")],
                        f"All block sizes / slab={slab} / {analysis}",root/"overview"/f"slab{slab:03d}__{analysis}__all_blocks_ranks_metrics.png")
            quality_chart(subset,f"SVD reconstruction: slab={slab}, {analysis}",
                          root/"overview"/f"slab{slab:03d}__{analysis}__psnr_vs_rank_all_blocks.png")
    write_pivot_tables(all_rows,root)
    write_guide(root)
    write_cross_contact_sheets(root)
    write_template_overviews(root)
    write_image_index(root)
    config.update(status="complete",configuration_count=len(configs),volume_metric_rows=len(all_rows),
                  elapsed_seconds=time.perf_counter()-started)
    (root/"manifest.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    print(f"Completed {len(configs)} configurations, {len(all_rows)} volume rows; {time.perf_counter()-started:.1f}s. {root}",flush=True)


def write_pivot_tables(rows, root):
    for analysis in ("raw","axial_centered"):
        for value in ("psnr_db","scalar_compression_ratio","uint8_source_to_float32_factor_ratio"):
            pivot={}
            for row in rows:
                if row["analysis"]!=analysis:
                    continue
                key=(row["configuration"],row["blocksize"],row["slab"])
                pivot.setdefault(key,dict(configuration=key[0],blocksize=key[1],slab=key[2]))[f"rank_{row['requested_rank']:03d}"]=row[value]
            write_csv(root/"tables"/f"{analysis}__{value}__block_slab_by_rank.csv",list(pivot.values()))


def write_guide(root):
    (root/"READ_FIRST.md").write_text("""# SVD 汇报输出阅读顺序

1. `cross_comparison.csv`：全部参数组合；这是原数据重建的参数对照，不是留出集交叉验证。
2. `tables/`：按 rank 展开列的 PSNR、标量比、uint8/fp32 理论字节比。
3. `overview/`：Pillow 直接绘制的 PNG 数值表、精度曲线、同切片不同blocksize重建/误差拼图和模板对照。
4. `configurations/<block或globalXY>__slab.../<raw或axial_centered>/`：每种配置。
5. `rank.../`：每个切片的原始尺寸重建 PNG、统一色标的绝对误差 PNG，以及原图/重建/误差拼图。
6. `templates/`：典型、P90、最困难完整块的二维模板；原始尺寸与256像素预览；轴向曲线与系数CSV。
7. `image_index.csv`：全部PNG的相对路径、图像尺寸、文件大小。

`globalXY` 表示整幅横向图像共享模板；slab等于总体层数时才是完整全局SVD。
`axial_centered` 的重建加回每个块在当前slab内的平均图，容量也计入均值图；rank不包含平均图。
误差图为逐像素绝对误差，不将局部PSNR误称为像素PSNR；PSNR数值按整切片或全体计算。
所有误差图使用同一个error-vmax；超出色标的误差饱和显示。黑色是零误差。
PSNR在未裁剪浮点重建上计算；输出PNG仅为裁剪到[0,1]并量化成8-bit的预览。
模板红正蓝负白零，每个模板独立对称归一化；比较强度必须参考sigma和系数CSV。
标量压缩比=原始体素数/因子标量数；uint8/fp32字节比=原始8-bit体字节数/(因子数*4)。
这是估计的因子容量，不包括文件头、分块索引和熵编码，也不代表Gaussian/Gabor模型大小。
PSNR=inf表示完整重建；接近完整秩的行不能作为低秩压缩成功证据。
尾slab保留实际层数，因此其实际rank可能小于requested_rank。
同切片拼图中，各配置使用相同切片编号；模板对照则各自选取中位误差示例，采样位置可能不同，坐标写在图中。
""",encoding="utf-8")


def write_cross_contact_sheets(root):
    """Compare all block sizes at exactly the same z, rank, slab and analysis."""
    root=Path(root)
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    selected=manifest["selected_slices"]
    z=selected[len(selected)//2]
    with (root/"cross_comparison.csv").open(encoding="utf-8-sig") as handle:
        rows=list(csv.DictReader(handle))
    groups={}
    for row in rows:
        groups.setdefault((row["analysis"],int(row["slab"]),int(row["requested_rank"])),[]).append(row)
    for (analysis,slab,rank),subset in groups.items():
        if rank==0:
            continue
        rec_images,err_images,labels=[],[],[]
        for row in subset:
            prefix=f"{row['configuration']}__{analysis}__rank{rank:03d}"
            base=root/"configurations"/row["configuration"]/analysis/f"rank{rank:03d}"
            rec=base/f"{prefix}__z{z:03d}__reconstruction_native.png"
            err=base/f"{prefix}__z{z:03d}__absolute_error_fixedscale_native.png"
            if not rec.exists() or not err.exists():
                continue
            with Image.open(rec) as image:
                rec_images.append(image.copy())
            with Image.open(err) as image:
                err_images.append(image.copy())
            labels.append(f"block={row['blocksize']}; rank={rank}\nVolume PSNR={float(row['psnr_db']):.3f} dB")
        if not rec_images:
            continue
        title=f"Same slice z={z} / slab={slab} / rank={rank} / {analysis}"
        stem=f"all_blocks__slab{slab:03d}__{analysis}__rank{rank:03d}__z{z:03d}"
        with Image.open(root/"originals"/f"original__z{z:03d}.png") as image:
            original=image.copy()
        sheet([original]+rec_images,["Original"]+labels,title,
              root/"overview"/f"{stem}__reconstructions_contact.png",columns=3)
        sheet(err_images,labels,title+f"\nAbsolute error; shared vmax={manifest['error_vmax']}",
              root/"overview"/f"{stem}__errors_contact.png",columns=3)


def write_image_index(root):
    root=Path(root)
    rows=[]
    for path in sorted(root.rglob("*.png")):
        with Image.open(path) as image:
            width,height=image.size
        rows.append(dict(path=path.relative_to(root).as_posix(),width=width,height=height,
                         bytes=path.stat().st_size))
    write_csv(root/"image_index.csv",rows)


def write_template_overviews(root):
    root=Path(root)
    with (root/"cross_comparison.csv").open(encoding="utf-8-sig") as handle:
        rows=list(csv.DictReader(handle))
    groups={}
    for row in rows:
        groups.setdefault((row["analysis"],int(row["slab"])),{})[row["configuration"]]=row["blocksize"]
    for (analysis,slab),configs in groups.items():
        images,labels=[],[]
        for config,blocksize in configs.items():
            folder=root/"configurations"/config/analysis/"templates"
            with (folder/"selected_blocks.csv").open(encoding="utf-8-sig") as handle:
                chosen=list(csv.DictReader(handle))
            if not chosen:
                continue
            block=chosen[0]
            prefix=f"{config}__{analysis}__block{int(block['block_index']):06d}__"
            paths=[]
            if analysis=="axial_centered":
                paths+=sorted(folder.glob(prefix+"*__mean_template_native.png"))
            paths+=sorted(folder.glob(prefix+"*__spatial_signed_native.png"))
            for i in range(4):
                if i<len(paths):
                    with Image.open(paths[i]) as image:
                        images.append(image.copy())
                    label="mean" if "mean_template" in paths[i].name else f"mode {i+1-int(analysis=='axial_centered')}"
                    labels.append(f"block={blocksize}; {label}\nxy=({block['x0']},{block['y0']}), z=[{block['z0']},{block['z1']})")
                else:
                    images.append(Image.new("RGB",(256,256),"white"))
                    labels.append("No additional independent mode")
        if images:
            sheet(images,labels,f"Templates across block sizes / slab={slab} / {analysis}\nPer-configuration median-error examples; positions differ. Signed templates independently normalized.",
                  root/"overview"/f"all_blocks__slab{slab:03d}__{analysis}__templates_contact.png",columns=4,nearest=True)


if __name__=="__main__":
    main()
