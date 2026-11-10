# 高斯柱修改说明：测试、真实数据与实验结果

本文对应本次 Gaussian 柱训练接入，记录测试代码在检查什么、真实数据如何准备、结果应该怎样解读。训练流程见 [COLUMN_TRAINING_FLOW.md](COLUMN_TRAINING_FLOW.md)，表示与保存结构见 [COLUMN_REPRESENTATION.md](COLUMN_REPRESENTATION.md)。

目前已经验证：共享 **XY footprint、沿 Z 轴排列分量**的柱可以接收梯度、拟合已知合成密度，并在原始渲染监督训练流程中更新和恢复。这里没有实现整根柱只占一个 BVH 基元，也没有完成与独立 3D Gaussian 的等质量、等参数量对照。

## 1. 新增测试文件及其作用

| 文件 | 输入与监督 | 主要回答的问题 |
|---|---|---|
| [test_gaussian_column.py](gfields/tests/test_gaussian_column.py) | 一根柱、两个轴向高斯，人为构造中心误差 | 两个分量的梯度会不会汇总到同一个横向参数？ |
| [test_gaussian_column_fit.py](gfields/tests/test_gaussian_column_fit.py) | 已知参数生成的密度，直接比较空间点密度 | 共享横向参数与独立轴向参数能否一起优化，恢复目标柱？ |
| [test_column_training.py](gfields/tests/test_column_training.py) | 小体数据、真实 Mitsuba 渲染、检查点 | 柱初始化、渲染梯度和保存恢复是否正确？同时提供原生 LS 子体准备入口。 |

三个脚本分别检查参数共享、密度拟合、渲染训练接入。前两个不测试 BVH 柱求交；第三个使用现有展开高斯渲染路径。

## 2. 最小共享梯度测试

源文件：[test_gaussian_column.py](gfields/tests/test_gaussian_column.py)。只存一个 `column_center_x`，轴向却有两个中心：

```python
column_optimizer["column_center_x"] = mi.Float([0.0])
column_optimizer["primitive_center_z"] = mi.Float([-0.4, 0.4])
gaussian_columns = GaussianColumn(column_optimizer, primitives_per_column=2)
centers, scales, quaternions, primitive_opacity = gaussian_columns.expand()
```

展开后两个高斯的 X 都为 0，Z 分别为 -0.4 和 0.4。测试还断言了 Y、横向尺度和旋转相同。随后故意给两个分量设置不同的 X 目标：

```python
target_centers_x = mi.Float([0.5, 1.0])
loss = dr.sum(dr.square(centers.x - target_centers_x))
dr.backward(loss)
shared_gradient = dr.grad(column_optimizer["column_center_x"]).numpy()
np.testing.assert_allclose(shared_gradient, [-3.0])
```

两个误差项对公共 X 的导数分别为 -1 和 -2，最后应累加为 -3。一次 Adam 更新后，公共 X 约为 0.1，重新展开的两个 X 都约为 0.1。测试检查 Z、尺度、旋转和权重没有被这个仅依赖 X 的损失改变。

这说明 `expand()` 的参数复制仍在自动微分图中。它不只是复制数值，而是能把多个展开分量的梯度送回同一个原始参数。

```powershell
.\.venv\Scripts\python.exe -m gfields.tests.test_gaussian_column
```

## 3. 已知目标的密度拟合测试

源文件：[test_gaussian_column_fit.py](gfields/tests/test_gaussian_column_fit.py)。目标与待拟合对象都是一根包含两个轴向高斯的柱，但初始中心、尺度、旋转和权重均有偏差。`evaluate_density()` 复用现有 `GaussianKernel.pdf()`，将各分量的密度相加：

```python
centers, scales, quaternions, primitive_opacity = gaussian_columns.expand()
density = mi.Float(0.0)
```

函数接下来逐个提取展开后的分量，构造 `Ellipsoid`，计算 `weight * gaussian_kernel.pdf(...)` 并累加。训练只比较空间采样点上的密度：

```python
dr.clear_grad(column_optimizer.variables)
predicted_density = evaluate_density(fitted_columns, training_points, gaussian_kernel)
loss = dr.mean(dr.square(predicted_density - target_training_density))
dr.backward(loss)
column_optimizer.step()
```

默认使用固定随机种子 `20261109`，32,768 个训练点、另取 16,384 个验证点，训练 500 步；`48×64×64` 规则网格只用于结果检查和出图。这里没有光线、图像监督或 OptiX 遍历。

测试不只要求损失下降，还检查八组参数的首步梯度均非零，以及中心、横向协方差、排序后的轴向中心、轴向尺度和权重是否接近目标。用协方差检查横向椭圆，是为了避免等价的轴交换和旋转参数写法影响判断。

已保存的 [metrics.json](results/gaussian_column_fit_results/metrics.json) 显示：

| 指标 | 结果 |
|---|---:|
| 初始规则网格 PSNR | 23.0736 dB |
| 最终规则网格 RMSE | 0，按本次 float32 计算 |
| 独立验证点 RMSE | 0，按本次 float32 计算 |
| 横向协方差相对误差 | 约 `4.89e-8` |
| `passed` | `true` |

误差为零时，JSON 中 `final_grid_psnr_db` 写作 `null`，并通过 `zero_grid_error_at_float32_precision: true` 说明原因。它不是一次失败或缺失结果；也不意味着连续实数意义上的所有位置误差严格为零。

```powershell
.\.venv\Scripts\python.exe -m gfields.tests.test_gaussian_column_fit
```

输出包括 [切片对比](results/gaussian_column_fit_results/density_fit__target_initial_fitted_errors.png)、[损失曲线](results/gaussian_column_fit_results/training_mse__log_scale.png)、[目标与拟合参数](results/gaussian_column_fit_results/parameters.json)、`loss_history.csv` 和原生切片 PNG。出图使用 Pillow。这个合成目标本来就属于柱模型能表达的函数族，因此它验证实现链路，不代表真实台风数据能达到相同误差。

## 4. 原始渲染流程的三个集成测试

源文件：[test_column_training.py](gfields/tests/test_column_training.py)，使用标准库 `unittest`，不要求安装 pytest：

```powershell
.\.venv\Scripts\python.exe -m gfields.tests.test_column_training
```

### 4.1 初始化与体素尺度约束

`test_shared_footprint_and_positive_initialization` 检查展开数量、组内 XY 参数共享，以及正尺度和正权重。还额外创建 `(Z,Y,X)=(8,12,16)` 的非立方体，在世界尺寸为各轴 2 的配置下检查半体素下界：

```python
anisotropic_volume = np.ones((8, 12, 16), dtype=np.float32)
columns = generate_random_columns(
    self.arguments.column_count, self.arguments.primitives_per_column, self.arguments.init_opacity,
    self.arguments.init_scale, self.arguments.init_albedo,
    volume=anisotropic_volume, volume_config=GenericConfig(),
    column_slab_depth=self.arguments.column_slab_depth
)
for name, half_voxel in zip(("column_scale_x", "column_scale_y", "primitive_scale_z"), (1.0 / 16, 1.0 / 12, 1.0 / 8)):
    self.assertAlmostEqual(columns.column_optimizer.bounds[name][1], half_voxel)
    self.assertTrue(np.all(columns.column_optimizer[name].numpy() >= half_voxel))
```

这能发现把 `(Z,Y,X)` 误当成 `(X,Y,Z)`、导致轴向尺度下界取错的问题。

### 4.2 渲染梯度与检查点恢复

`test_render_gradient_and_raw_checkpoint_round_trip` 从 `create_scene_dict_from_primitives(..., use_columns_representation=True, gaussian_columns=gaussian_columns)` 创建现有场景，将原始柱参数写入场景后实际渲染：

```python
self.write_optimizer_params_to_scene(columns.column_optimizer, scene_params, level=0, gaussian_columns=columns)
image = mi.render(scene, scene_params, spp=2, seed=17)
weighted_pixels = 1.0 + mi.Float(np.linspace(0.0, 1.0, dr.width(image.array), dtype=np.float32))
loss = dr.mean(dr.square(image.array) * weighted_pixels)
dr.backward(loss)
```

像素权重用于构造有空间差异的诊断损失。测试断言共享 `column_center_x` 和各分量 `primitive_opacity` 的梯度有限且非零，并检查优化器更新后权重确实变化。

同一个测试保存并加载原始柱检查点，断言参数、展开结果、分量数量、迭代号和约束全部一致。随后让保存前和恢复后的两个优化器执行相同的一次更新：

```python
for current_optimizer in (optimizer, restored.column_optimizer):
    dr.clear_grad(current_optimizer.variables)
    dr.backward(dr.sum(dr.square(current_optimizer["primitive_opacity"])))
    current_optimizer.step()
```

最后逐项比较更新后的参数完全相同。这说明恢复的不只有数值，也包括影响下一次更新的 Adam 状态。这个单步一致性测试不等同于允许任意更改相机、SPP 或学习率计划后仍保持整次训练逐步相同。

### 4.3 缺少共享参数时拒绝恢复

`test_expanded_asset_without_column_sidecar_is_rejected` 创建只有展开高斯文件、没有 `columns.npz` 的目录，要求 `load_columns()` 抛出 `FileNotFoundError`。展开高斯虽然可以渲染，却不足以唯一说明哪些分量原本共享一个 footprint，不能悄悄作为柱检查点继续训练。

上述三个集成测试已经通过。测试临时目录放在项目 `results/column_test_results/` 下，避免依赖系统临时目录的写入权限。

## 5. 真实 LS 子体如何生成

同一个集成测试模块提供了数据准备模式：

```powershell
.\.venv\Scripts\python.exe -m gfields.tests.test_column_training --prepare-real-data
```

默认读取 `data/ls_data/22_cc_Area_*.png`，按文件名末尾的数字排序，要求原始 30 张切片。先把切片归一化到 0–1，再计算沿 Z 的平均图像。在不重叠的 64×64 窗口中选择平均投影方差最大的区域，目的是挑选有明显结构变化的样本。

核心裁剪、堆叠和补零步骤为：

```python
slices = [np.asarray(Image.open(path).convert("L").crop((left, top, left + block_size, top + block_size)), dtype=np.float32) / 255.0 for path in slice_paths]
native_volume = np.stack(slices)
padding = (block_size - len(slices)) // 2
volume = np.pad(native_volume, ((padding, block_size - len(slices) - padding), (0, 0), (0, 0)))
```

本次 [crop_metadata.json](results/gaussian_column_training_results/crop_metadata.json) 记录：

| 项目 | 数值 |
|---|---|
| NumPy 轴顺序 | `(Z,Y,X)` |
| X 裁剪范围 | `[448,512)` |
| Y 裁剪范围 | `[768,832)` |
| 原生子体形状 | `(30,64,64)` |
| Z 补零 | 前后各 17 层 |
| 提交训练的形状 | `(64,64,64)` |
| 原生层在补零体中的范围 | `z=17..46`，即 Python 切片 `[17:47]` |
| 是否插值 | 否 |

选区来自全 30 层平均图案的局部方差，并非人工指定了最好拟合的单张切片。补零让原渲染流程使用立方网格，同时保留子体内原生体素间距和 `30/64` 的厚度比例。这是小型真实数据验证，不覆盖原始全场的全部结构。

生成文件为 [ls_native30_crop64_cube.npy](results/gaussian_column_training_results/ls_native30_crop64_cube.npy)，代表切片预览为 [native_crop_z015.png](results/gaussian_column_training_results/native_crop_z015.png)。这里文件名的 15 是原生层索引，对应补零后 `z=32`。

## 6. 两组最终真实数据实验

完整训练命令见根目录 [README.md](README.md) 的 Shared-footprint Gaussian columns 部分。两组共同配置如下：

| 项目 | 配置 |
|---|---|
| 训练入口 | 原有 `python -m gfields.train` |
| 模式 | `--use_columns_representation`，Gaussian 阶段，不训练 Gabor |
| 总柱数 | 64，分配到初始化 slabs，而非每个 slab 各 64 根 |
| 每柱分量 | 4，总共展开 256 个 Gaussian |
| 迭代数 | 200 |
| 相机 | 16 个，32×32 像素 |
| 参考图像 | 64 SPP |
| 优化 SPP / 梯度 SPP | 2 / 2 |
| 最终评估渲染 | 16 SPP，seed=0；初始与最终一致 |
| 密度倍数 | `--opacity_scale 1` |
| 基础学习率 | `--column_lr 0.003` |
| 初始权重 | `--init_opacity 0.003` |
| 尺度下界 | 对应世界体素间距的一半 |

8 层和 16 层只改变初始化时的轴向分组。训练后参数可以跨越初始化分组，尾部也不会按 slab 裁断。

下表来自 [experiment_summary.csv](results/gaussian_column_training_results/experiment_summary.csv)，为便于阅读保留四位小数；JSON/CSV 中保存完整数值。

| 初始化 slab | 渲染 PSNR 初始 → 最终 | 原生 30 层区域密度 PSNR 初始 → 最终 | 补零全体密度 PSNR 最终 |
|---|---:|---:|---:|
| 8 层 | 13.5437 → 31.3957 dB | 7.8401 → 15.7271 dB | 18.9600 dB |
| 16 层 | 13.9086 → 31.8597 dB | 7.9426 → 15.9598 dB | 19.1896 dB |

渲染 PSNR 比较的是**训练相机的透射图像**，不是未见视角；原生区域密度 PSNR 比较的是全部原生 Z 层及其完整 64×64 XY 区域。补零全体指标还纳入前后空白区域，不能替代原生区域指标。

## 7. 参数量和输出文件该怎样看

64 根柱、每柱 4 个轴向分量共有：

```text
每柱共享参数：column_center_x, column_center_y, column_scale_x, column_scale_y, column_quaternions → 5
每个轴向分量：primitive_center_z, primitive_scale_z, primitive_opacity          → 3
标量总数：64 * (5 + 3 * 4) = 1088
float32 参数字节数：1088 * 4 = 4352 bytes = 4.25 KiB
```

4,352 字节指训练参数数组的裸数值大小，**不是 `columns.npz` 的文件大小，也不是完整渲染资产大小**。检查点还存 Adam 状态、迭代号、边界和元数据；现有渲染器还需要 256 个展开高斯的数据。

| 文件 | 应该用于什么 |
|---|---|
| `column_render_metrics.json` | 初始、最终渲染图像的 MSE、PSNR、SSIM、L1；参数统计与已完成步数 |
| `column_training_metrics.csv` | 训练过程中记录的损失和图像指标；此处 `SSIM` 列为训练 SSIM 损失，越低越好 |
| `column_slices/final__metrics.json` | 不裁剪数值的解析密度误差；全体、原生 Z 区间和各 Z 切片指标 |
| `column_slices/final__slice_comparison.png` | 三个代表 Z 位置，每行依次为目标、重建、绝对误差 |
| `optimized_asset_pyr0/npy_data/columns.npz` | 恢复共享训练关系的原始参数和 Adam 状态 |
| `optimized_asset_pyr0/npy_data/column_metadata.json` | 可以直接阅读的参数数量、下界和迭代号 |
| `optimized_asset_pyr0/data/` | 供现有渲染器读取的展开资产 |

`column_render_metrics.json` 中的 `ssim` 是 `1 - SSIMLoss`，越高越好，和 CSV 列的方向相反。查看终端日志中的历史 `Final SSIM` 输出时，也应注意原有打印调用的是损失函数。

密度图与误差图采用统一显示范围 0–1，超出显示范围的像素会饱和，但 JSON 中的 MSE/PSNR 按未裁剪重建数值计算。密度评估包含解析高斯尾部；渲染器仍使用有限椭球包围区域，因此两类指标评估对象不同。

先查看下面两张图，再结合对应 JSON 读数：

- [8 层初始化：最终切片对比](results/gaussian_column_training_results/slab08_voxel_bound/column_slices/final__slice_comparison.png)
- [16 层初始化：最终切片对比](results/gaussian_column_training_results/slab16_voxel_bound/column_slices/final__slice_comparison.png)

重建能复现部分大尺度结构，但明显比原生切片平滑，细纹理仍有较大误差。现有结果没有证明高斯柱能无损替代原始高频结构。

## 8. 为什么保留了较早的 `slab08/` 结果

`results/gaussian_column_training_results/slab08/` 是较早的 4 相机、160 步试跑，尚未加入半体素尺度下界。它的渲染 PSNR 虽然上升，但部分轴向尺度收缩为亚体素尖峰，补零体 `z=46` 的密度 MSE 达到约 12.3213，切片 PSNR 约 -10.9066 dB。

这次检查直接促成了体素尺度下界的修改。当前结果应使用 `slab08_voxel_bound/` 与 `slab16_voxel_bound/`。旧目录保留用于解释问题，不能与最终目录混成同一组实验；相机数、步数和约束同时改变，也不能仅凭旧新差异定量归因于某一个改动。

另外，`resume_smoke/` 已从 8 层最终检查点恢复并完成第 201 步，记录的初始渲染 PSNR 与原实验最终值一致；`ordinary_smoke/` 只检查未开启柱模式的原始独立 Gaussian 路径能完成一次更新，不是公平的独立高斯质量基线。

## 9. 这次结果能回答什么，下一步还缺什么

已回答的问题是共享参数的自动微分、密度拟合、真实渲染梯度、检查点恢复和原训练流程接入是否可以工作。当前代码与测试支持这个判断。

尚未做的对照包括：同高斯数量下柱与独立 Gaussian 的质量、同参数字节数下的质量、多个随机种子和更多真实子体、相同质量下的渲染耗时，以及整柱 BVH 基元与展开高斯的求交性能。8 层与 16 层这一次约 0.23 dB 的原生密度 PSNR 差异，不足以确定最优共享深度。

如果只把同一份已经训练好的柱参数换成原生柱求交，并保持核函数、支持范围与积分语义一致，目标是保持同一个密度场与渲染质量，再比较遍历和计算时间。它不会因为更换存储组织而自动提高拟合 PSNR。
