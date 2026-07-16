# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Single-frame rendering and video (EXR sequence -> MP4) helpers."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional


def render_single_frame(scene, camera, *, spp: int, seed: int = 0):
    """Render *scene* through *camera* with `vp.benchmark.single_run` timing."""
    import mitsuba as mi
    from gfields import benchmark as vp_bench

    with vp_bench.single_run("Rendering"):
        return mi.render(scene, sensor=camera, spp=spp, seed=seed)


def compile_exr_paths_to_mp4(
    exr_paths: List[str],
    framerate: float,
    output_mp4: str,
    *,
    crf: int = 18,
) -> bool:
    """Encode a list of EXR frames into an H.264 MP4 via ffmpeg's concat demuxer."""
    if not exr_paths:
        return False
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found in PATH; skipping MP4 compile.")
        return False
    duration = 1.0 / float(framerate)
    fd, list_path = tempfile.mkstemp(suffix=".ffconcat", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("ffconcat version 1.0\n")
            for p in exr_paths:
                p_abs = os.path.abspath(p).replace("\\", "/")
                p_esc = p_abs.replace("'", "'\\''")
                f.write(f"file '{p_esc}'\n")
                f.write(f"duration {duration}\n")
            p_last = os.path.abspath(exr_paths[-1]).replace("\\", "/")
            p_last_esc = p_last.replace("'", "'\\''")
            f.write(f"file '{p_last_esc}'\n")
        cmd = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), output_mp4,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-2500:]
            print(f"ffmpeg failed (exit {proc.returncode}). Last output:\n{tail}")
            return False
        print(f"Compiled video: {output_mp4}")
        return True
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass


# Alias matching the README API table -- ``video.assemble(...)`` reads more
# naturally in thin CLIs than ``video.compile_exr_paths_to_mp4(...)``.
def assemble(frames: List[str], output: str, *, framerate: float = 24.0, crf: int = 18) -> bool:
    """Alias for :func:`compile_exr_paths_to_mp4` with a cleaner signature."""
    return compile_exr_paths_to_mp4(frames, framerate, output, crf=crf)


def frames_to_mp4(frames, output_path, *, fps: float = 30.0, crf: int = 18) -> bool:
    """Encode in-memory RGB frames into an H.264 MP4 by piping raw rgb24 to ffmpeg.

    ``frames`` is a list of ``HxWx3`` (or ``HxWx>=3``) numpy arrays; float frames
    in ``[0, 1]`` are scaled to uint8, uint8 frames are used as-is. Returns True
    on success. Shared by the video CLIs (``tornado`` / ``steerable``) which each
    used to carry their own near-identical ffmpeg encoder.
    """
    import numpy as np

    if not frames:
        return False
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found in PATH; skipping MP4 compile.")
        return False

    def _to_u8(f):
        a = np.asarray(f)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        return np.ascontiguousarray(a[..., :3])

    frames_u8 = [_to_u8(f) for f in frames]
    height, width = frames_u8[0].shape[:2]
    cmd = [
        ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", str(crf), str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames_u8:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode == 0:
        print(f"Video saved: {output_path}")
        return True
    print(f"ffmpeg returned code {proc.returncode}")
    return False


def render_video_frames(
    scene_dict_base: Dict[str, Any],
    cameras: List,
    cam_index: int,
    *,
    spp: int,
    output_config,  # rendering.pipeline.OutputConfig-like, kept Any to avoid circular import
    framerate: int,
    duration: int,
    level_idx: Optional[int] = None,
    compile_mp4: bool = True,
    video_crf: int = 18,
    envmap_rotation_x: float = 0.0,
    envmap_rotation_y: float = 135.0,
    envmap_rotation_z: float = 0.0,
) -> List[str]:
    """Render a rotating-light video for one camera and (optionally) compile to MP4.

    Returns the list of rendered EXR paths in order. Matches the legacy behavior
    of ``examples/render_asset_new.render_video_frames``.
    """
    import mitsuba as mi
    from tqdm import tqdm

    from .pipeline import envmap_to_world_transform

    total_frames = framerate * duration
    rendered_paths: List[str] = []
    for frame_idx in tqdm(range(total_frames), desc="Rendering video"):
        spin_y = 360.0 / total_frames * frame_idx
        scene_dict_base["environment"]["to_world"] = envmap_to_world_transform(
            spin_y, envmap_rotation_x, envmap_rotation_y, envmap_rotation_z
        )
        scene = mi.load_dict(scene_dict_base)
        frame = mi.render(scene, sensor=cameras[cam_index], spp=spp)
        output_path = output_config.get_frame_path(
            frame_idx=frame_idx, cam_idx=cam_index, level_idx=level_idx, suffix="video"
        )
        mi.util.write_bitmap(output_path, frame)
        rendered_paths.append(output_path)
        print(f"  Saved: {os.path.basename(output_path)}")

    if compile_mp4 and rendered_paths:
        if level_idx is not None:
            mp4_name = f"{output_config.asset_name}_rotating_light_level{level_idx}.mp4"
        else:
            mp4_name = f"{output_config.asset_name}_rotating_light.mp4"
        out_mp4 = os.path.join(output_config.experiment_dir, mp4_name)
        compile_exr_paths_to_mp4(rendered_paths, float(framerate), out_mp4, crf=video_crf)

    return rendered_paths


__all__ = [
    "render_single_frame",
    "compile_exr_paths_to_mp4",
    "assemble",
    "render_video_frames",
]
