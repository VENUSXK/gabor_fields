> Current setup: generate or load column parameters, create the scene, read scene parameters into a new optimizer, set bounds and learning rates, then restore checkpoint Adam state through load_columns(..., opt=opt). Column generation and checkpoint loading do not create an optimizer.

# Gaussian 柱的参数表示与保存逻辑

本文解释这次柱状表示在代码中具体怎么存、怎么展开、怎么初始化和恢复。训练入口见 [COLUMN_TRAINING_FLOW.md](COLUMN_TRAINING_FLOW.md)。

当前实现的是**共享 x-y footprint、沿 z 轴安排多个高斯分量**。它已经能在原始训练流程中优化；渲染时仍展开为普通高斯。整根柱作为一个 BVH 自定义基元的实现尚未加入。

## 1. 本次涉及哪些代码

| 文件 | 本次柱模式相关内容 |
|---|---|
| [gfields/integrators/common.py](gfields/integrators/common.py) | 之前加入的 `GaussianColumn` 类；复用已有 `Ellipsoid.ravel()` |
| [gfields/training/columns.py](gfields/training/columns.py) | 新增的初始化、优化器、场景初值、参数统计、检查点、密度切片工具 |

`GaussianColumn` 是“共享参数到现有高斯场景的适配器”。它没有实现新的 Gaussian 核，也没有实现新的 OptiX 形状。

## 2. 一根柱里有哪些参数

设有 `M` 根柱，每根柱有 `K` 个轴向高斯。原始可训练参数直接放在 `BoundedAdam` 中，分成 8 组一维 `mi.Float` 数组：

| 键名 | 数组长度 | 表示什么 | 同一根柱内是否共享 |
|---|---:|---|---|
| `column_center_x` | M | footprint 的 x 中心 | 是 |
| `column_center_y` | M | footprint 的 y 中心 | 是 |
| `column_scale_x` | M | footprint 的横向尺度 | 是 |
| `column_scale_y` | M | footprint 的另一横向尺度 | 是 |
| `column_quaternions` | M | footprint 在 x-y 平面内的旋转角，单位弧度 | 是 |
| `primitive_center_z` | M×K | 每个分量的 z 中心 | 否 |
| `primitive_scale_z` | M×K | 每个分量的 z 尺度 | 否 |
| `primitive_opacity` | M×K | 每个分量的正权重 | 否 |

这意味着一根柱内部的 K 个高斯投影到 x-y 平面后，中心、椭圆形状、旋转完全相同，但在 z 上的位置、厚度、权重可以不同。当前没有倾斜整根柱的 x/y 旋转自由度。

例如两根柱、每根三个分量：

```text
共享参数：       第0根柱             第1根柱
column_center_x：      [x0,                x1]
轴向参数顺序：  [z00, z01, z02,     z10, z11, z12]
```

代码中的名称集中定义在 `columns.py` 顶部：

```python
parameter_names = ('column_center_x', 'column_center_y', 'column_scale_x', 'column_scale_y', 'column_quaternions', 'primitive_center_z', 'primitive_scale_z', 'primitive_opacity')
```

固定 `albedo` 作为 `columns.init_albedo` 保存，不在这 8 组优化参数里；Gaussian 阶段的 `omega` 写为零。

## 3. `expand()` 怎么把共享参数变成普通高斯

位置：`common.py` 中的 `GaussianColumn.expand()`。

首先给全部 M×K 个分量编号，再用整数除法求出它属于哪根柱：

```python
primitive_indices = dr.arange(mi.UInt32, primitive_count)
column_indices = primitive_indices // self.primitives_per_column
```

对于 `M=2, K=3`：

```text
primitive_indices = [0, 1, 2, 3, 4, 5]
column_indices    = [0, 0, 0, 1, 1, 1]
```

随后 `dr.gather()` 按这些柱编号取出共享参数：

```python
column_center_x = dr.gather(mi.Float, column_optimizer["column_center_x"], column_indices)
column_center_y = dr.gather(mi.Float, column_optimizer["column_center_y"], column_indices)
column_scale_x = dr.gather(mi.Float, column_optimizer["column_scale_x"], column_indices)
column_scale_y = dr.gather(mi.Float, column_optimizer["column_scale_y"], column_indices)
column_quaternions = dr.gather(mi.Float, column_optimizer["column_quaternions"], column_indices)
```

于是 `column_center_x=[x0,x1]` 在渲染侧变为 `[x0,x0,x0,x1,x1,x1]`。但优化器仍然只保存两个 x 值。

这些操作保留自动微分关系。因此，同一根柱内部三个分量对 x 中心产生的梯度，会累加到同一个 `column_center_x` 元素上。它们不会各自获得一个独立的可训练 x 中心。

最后组合成现有渲染器所需的数据：

```python
centers = mi.Point3f(column_center_x, column_center_y, column_optimizer["primitive_center_z"])
scales = mi.Vector3f(column_scale_x, column_scale_y, column_optimizer["primitive_scale_z"])
quaternions = dr.euler_to_quat(mi.Point3f(0.0, 0.0, column_quaternions))
primitive_opacity = column_optimizer["primitive_opacity"]
return centers, scales, quaternions, primitive_opacity
```

这里前两轴来自共享数组，第三轴来自独立轴向数组；旋转只有 z 轴角度，即 x-y 平面旋转。

## 4. `write_optimizer_params_to_scene()` 与已有 `Ellipsoid` 的关系

位置：`common.py` 中的 `GaussianColumn.write_to_scene()`。核心逻辑如下：

```python
centers, scales, quaternions, primitive_opacity = self.expand()
primitive_name = f"primitives_pyr{level}"
scene_params[f"{primitive_name}.data"] = Ellipsoid.ravel(centers, scales, quaternions)
scene_params[f"{primitive_name}.opacities"] = primitive_opacity
scene_params[f"{primitive_name}.omega"] = dr.zeros(mi.Float, dr.width(primitive_opacity))
scene_params.update()
```

已有的 `Ellipsoid.ravel()` 把每个高斯的几何属性打包成 10 个浮点数：

```text
中心3个 + 尺度3个 + 四元数4个
[cx, cy, cz, sx, sy, sz, qx, qy, qz, qw]
```

权重通过 `opacities` 属性另外传入。此处复用了原来的数据格式，所以现有 Gaussian/Gabor 核和渲染器可以继续读取。

`write_optimizer_params_to_scene()` 没有给 BVH 提交“一根柱”的新几何类型。当前场景创建仍使用 `ellipsoidsmesh`，展开后的每个高斯由原来的三角形壳参与求交。

`create_scene_dict_from_primitives(..., use_columns_representation=True, gaussian_columns=gaussian_columns)` 则用于**创建场景的初始字典**：它把展开结果转为 `mi.TensorXf`。这一步有 `.numpy()`，不用于保持每轮训练的梯度链；场景创建后，通过可微的 `write_optimizer_params_to_scene()` 重新接到共享参数。

## 5. 初始 footprint 和轴向分量怎么生成

位置：`columns.py` 中的 `generate_random_columns()`。

输入数组必须是 `(Z,Y,X)` 顺序的有限三维数组，`opacity_scale` 必须大于零。当前仅接受轴对齐的体变换。

### 5.1 先找到有效 z 区间，再均匀分 slab

```python
occupied = np.any(np.abs(volume) > 1e-6, axis=(1, 2))
occupied_z = np.flatnonzero(occupied)
start_z, end_z = (int(occupied_z[0]), int(occupied_z[-1]) + 1) if occupied_z.size else (0, volume.shape[0])
slab_depth = args.column_slab_depth or end_z - start_z
slab_starts = np.arange(start_z, end_z, slab_depth)
```

例如 30 层有效数据补零到 64³ 后，有效 z 是 `[17,47)`。设置 8 层 slab 时，初始化分组是 `[17,25)`、`[25,33)`、`[33,41)`、`[41,47)`；最后一组只有 6 层。

`--column_count` 是**所有 slab 加起来的柱总数**。代码轮流给各 slab 分配柱：

```python
slab_indices = np.arange(column_count) % len(slab_starts)
```

64 根柱分到 4 个 slab，就是每个 slab 16 根。当前不同 slab 的 footprint 各自初始化和优化，**没有额外实现“相邻 slab 之间共享同一套二维字典”**。

### 5.2 在每个 slab 内按密度投影采样 x-y 中心

```python
projection = np.maximum(volume[int(slab_start):min(int(slab_start) + slab_depth, end_z)], 0.0).sum(axis=0, dtype=np.float64)
probabilities = projection.reshape(-1)
total = probabilities.sum()
slab_projections[slab_index] = (projection.shape, probabilities / total if total > 0.0 else None)
```

投影中密度较高的位置，被选为 footprint 中心的概率较大。全零 slab 使用均匀采样。索引随后通过 `volume_config.to_world` 转到世界坐标；采样位置对应体素中心。

初始 `column_scale_x` 与 `column_scale_y` 相等，大小按每个 slab 的柱数和 `init_scale` 估计；角度随机生成。训练后两轴尺度可以不同，footprint 可以成为椭圆。

### 5.3 在 slab 内均匀放置 K 个 z 中心

```python
normalized_z = (lower + (np.arange(primitive_count) + 0.5) * (upper - lower) / primitive_count) / volume.shape[0]
primitive_center_z.extend(matrix[2, 2] * normalized_z + matrix[2, 3])
primitive_scale_z.extend([max(1e-4, abs(matrix[2, 2]) * (upper - lower) / volume.shape[0] / primitive_count * 0.65)] * primitive_count)
```

以 8 层、4 个分量为例，初始分量中心位于该 slab 内约第 1、3、5、7 层的位置，初始尺度约为分量间距的 0.65 倍。权重统一来自 `--init_opacity`。

**slab 只用于初始化。** 后续没有把分量锁在所属 slab，也没有在 slab 边界把高斯尾部清零。轴向中心可以在全局中心约束内更新，尾部也会跨越 slab 边界参与表示。

## 6. 原优化器怎么优化这些数组

位置：`columns.py` 中的 `write_scene_params_to_column_optimizer()`。

```python
optimizer = gfields.optimizers.BoundedAdam(lr=args.column_lr, noise_init=0.0)
```

仍使用项目已有的 `BoundedAdam`。这里关闭旧的独立基元噪声逻辑，因为它按 `centers_0`、`scales_0` 等旧键名组织，不适用于新的 8 组参数。

各组学习率为 `column_lr` 乘以已有倍率：中心使用 `centers_lr`，尺度使用 `scales_lr`，平面角度使用 `quats_lr`，权重使用 `opacity_lr`。

| 参数 | 当前边界 |
|---|---|
| `column_center_x`、`column_center_y`、`primitive_center_z` | `[-1,1]` |
| 三组尺度 | 对应轴半体素世界间距至 `1.0`，最小不小于 `1e-5` |
| `primitive_opacity` | `[1e-8,1.0]` |
| `column_quaternions` | 没有显式边界 |

半体素尺度下界在初始化时计算：

```python
world_voxel_size = np.abs(np.diag(matrix)[:3]) / np.asarray(volume.shape[::-1])
scale_lower_bounds = dict(zip(('column_scale_x', 'column_scale_y', 'primitive_scale_z'), np.maximum(0.5 * world_voxel_size, 1e-5)))
columns = GaussianColumn(parameters, primitives_per_column, scale_lower_bounds)
```

例如 64³ 体在世界中占据边长 2 的立方体时，体素间距是 `2/64`，尺度下界是 `1/64=0.015625`。这是为了抑制远小于体素的密度尖峰；它是本次实验使用的约束选择，不是柱表示必然要求的数学条件。

## 7. 密度公式与参数量

当前柱的连续密度可写为：

\[
\kappa(x,y,z)=\sum_{i=1}^{M}g_i^{xy}(x,y)\sum_{j=1}^{K}w_{ij}g_{ij}^{z}(z).
\]

这里 `g_i^{xy}` 是积分为 1 的二维 Gaussian，`g_ij^z` 是积分为 1 的一维 Gaussian。两者相乘就是现有归一化三维 Gaussian 的 `pdf`。所以 `primitive_opacity` 代表乘在归一化核上的权重，不是高斯峰值；尺度变化时，即使权重不变，峰值也会变化。

密度切片工具将重建的 `kappa` 除以 `volume_config.opacity_scale`，再与原始体素值比较，保证二者处于同一数值尺度：

```python
reconstructed += footprints[column_index] * profile / volume_config.opacity_scale
```

密度诊断计算完整解析尾部；现有渲染器使用有限几何包围壳筛选参与积分的高斯，二者在支持范围附近可能有差异。因此密度 PSNR 与投影渲染 PSNR 是两个不同的指标。

一根柱有 5 个共享标量、每个分量有 3 个独立标量，因此：

\[
N_{\mathrm{params}}=M(5+3K),\qquad \mathrm{bytes}=4M(5+3K).
\]

64 根柱、每柱 4 个分量时为 1,088 个 float32 标量，即 4,352 字节。这个数字只计算可训练原始参数，不包含 Adam 状态、元数据、GPU 中展开的数据或导出的完整场景。

`GaussianColumn.parameter_stats()` 同时给出对照计数：每个独立的相同旋转约束高斯是 8 个标量；每个完整 3D 高斯按中心 3、尺度 3、四元数 4、权重 1 计为 11 个标量。计数差异不等于已经完成同等参数预算的质量对比。

## 8. 为什么必须另存 `columns.npz`

位置：`columns.py` 中的 `save_columns()`、`load_columns()`。

普通导出资产只知道展开后的 M×K 个高斯，无法仅从这些数值可靠地恢复柱分组和共享优化关系。因此训练检查点另存：

```text
optimized_asset_pyr0/
├── data/                    # 现有格式的展开渲染资产
└── npy_data/
    ├── columns.npz          # 共享参数和 Adam 状态
    └── column_metadata.json # 可直接阅读的元数据
```

保存时先取出原始 8 组数组：

```python
optimizer = columns.column_optimizer
arrays = {name: np.asarray(optimizer[name], dtype=np.float32) for name in parameter_names}
scale_lower_bounds = {name: optimizer.bounds[name][1] for name in ('column_scale_x', 'column_scale_y', 'primitive_scale_z')}
```

同时保存每组 Adam 的步数、一阶矩、二阶矩及存在时的最大二阶矩，并记录每柱分量数、完成迭代数、固定 albedo 和尺度下界。NPZ 先写为临时文件，再替换正式文件。

加载时先验证参数长度和有限性，再创建相同数组布局的优化器、恢复 Adam 状态，最后构造 `GaussianColumn`。每柱分量数来自检查点元数据；学习率仍来自本次命令行设置。

```python
columns = GaussianColumn(optimizer, metadata['primitives_per_column'])
columns.init_albedo = float(metadata.get('init_albedo', 0.0))
return columns, int(metadata.get('iteration', 0))
```

如果只有展开资产而没有 `columns.npz`，加载会明确抛出 `FileNotFoundError`，不会悄悄恢复成彼此独立的高斯。

恢复后保持的是同一套共享结构和 Adam 状态。若改变学习率、相机或目标总迭代数等训练设置，后续轨迹仍可能变化；检查点并没有冻结全部命令行配置。
