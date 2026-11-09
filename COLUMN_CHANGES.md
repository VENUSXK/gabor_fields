# Gaussian 柱状表示：本次修改说明

本文对应当前工作区的实现，记录日期为 2026-11-09。建议先读本文，再按下面的顺序阅读三个专题文档。文档中的代码片段用于解释现有实现；省略处不代表可以单独复制运行。

| 文档 | 要回答的问题 |
|---|---|
| [表示、初始化与检查点](COLUMN_REPRESENTATION.md) | 一根柱保存什么？怎样展开？梯度如何回到共享参数？ |
| [训练流程与运行方法](COLUMN_TRAINING_FLOW.md) | 原训练代码在哪些地方切换到柱模式？如何运行和恢复？ |
| [测试、输出与实验结果](COLUMN_TESTS_AND_RESULTS.md) | 怎样证明代码可用？图怎么看？结果能说明什么？ |

## 先说明现在实现到了哪一步

当前实现是：**训练时共享 x-y footprint，沿 z 轴有多个独立高斯分量，渲染时展开为现有 3D 高斯。**

共享的五个参数是 x 中心、y 中心、x 尺度、y 尺度和平面旋转角；每个轴向分量另有 z 中心、z 尺度和正权重。这里不是共享 x-z 平面。输入数组按 `(Z, Y, X)` 排列，而场景点使用 `(x, y, z)`，二者的顺序需要区分。

```text
原生体素 / Gaussian 目标体
        ↓
InitializeColumns：初始化共享 footprint 和轴向分量
        ↓
BoundedAdam：持有原始共享参数
        ↓
GaussianColumn.Expand：为渲染临时展开参数
        ↓
WriteToScene：写入现有 ellipsoidsmesh 场景
        ↓
原有可微透射渲染 → 图像损失 → 反向传播
        ↓
更新共享参数，下一轮重新展开
```

因此，这次完成了表示和训练接入；**尚未实现“一根柱只作为一个 BVH 基元”**。当前训练场景使用三角形壳 `ellipsoidsmesh`；展开后的高斯仍分别参与现有求交流程。不能把参数减少直接当作 BVH 求交次数或渲染时间减少。

## 修改了哪些文件

| 文件 | 本次柱模式的作用 |
|---|---|
| [common.py](gfields/integrators/common.py) | 前期新增 `GaussianColumn`，提供 `Expand()` 与 `WriteToScene()`；本次训练接入复用此类 |
| [columns.py](gfields/training/columns.py) | 新增柱初始化、优化器配置、场景初始参数转换、参数统计、原始柱检查点和密度切片诊断 |
| [regression.py](gfields/training/regression.py) | 新增命令行参数、柱模式校验及 Gaussian-only 配置 |
| [train.py](gfields/train.py) | 在原训练流程中选择柱初始化/加载、共享参数优化、渲染前更新和柱检查点保存 |
| [test_gaussian_column.py](gfields/tests/test_gaussian_column.py) | 前期的小测试：多个展开分量的梯度是否累加到同一个共享参数 |
| [test_gaussian_column_fit.py](gfields/tests/test_gaussian_column_fit.py) | 前期的合成密度拟合：已知目标参数是否能恢复 |
| [test_column_training.py](gfields/tests/test_column_training.py) | 新增训练接入测试，检查真实渲染梯度、约束和 Adam 恢复；同时提供原生 LS 小体准备命令 |
| [README.md](README.md) | 增补根目录环境下的命令、输出及边界说明 |

`common.py` 当前 diff 还包含已有的 `GaussianKernel.inv_cdf()` 处理变化；它不是本次柱训练接入新增的算法，不应和 `GaussianColumn` 一起归为本次改动。已有数据转换脚本、原始图片、完整体数据和 SVD 工具也不属于这次柱训练改动。

## 关键区别：优化器里保存谁

原模式优化的是每个高斯独立的中心、尺度和旋转。柱模式让优化器只保存一份横向参数，展开后多个高斯引用同一个来源：

```python
component_indices = dr.arange(mi.UInt32, component_count)
column_indices = component_indices // self.components_per_column
center_x = dr.gather(mi.Float, column_optimizer["center_x"], column_indices)
```

例如每柱 4 个分量，`column_indices` 就是 `[0, 0, 0, 0, 1, 1, 1, 1, ...]`。这使一根柱内部四个高斯的 x 坐标来自同一个可训练变量，反向传播时梯度也累加回同一个变量。

训练接入没有把展开结果重新登记成独立优化参数，而是使用：

```python
opt = gaussian_columns.column_optimizer
gaussian_columns.WriteToScene(scene_params, level=GAUSS_LVL)
```

场景只是接收展开结果，优化器仍更新原始柱参数。这是保持共享关系的关键。

## 为何增加尺度下界和第二份检查点

最初的小实验中，4 个相机的渲染 PSNR 提升，但部分轴向高斯缩小到远小于体素，产生密度尖峰。因此，柱初始化和优化都加入了对应轴半个世界体素间距的尺度下界；它是可调整的表示约束，并不保证恢复所有真实细节。

原有 `save_model()` 继续保存可供现有渲染器读取的展开资产。同时新增 `columns.npz` 保存共享参数、Adam 状态、迭代数、尺度下界和固定 albedo。只有展开高斯的资产不包含足够的共享信息，柱模式恢复时会拒绝它，而不是猜测哪些高斯应该归为一根柱。

初始化 slab 只决定起始分配和轴向分量的位置。没有给每个 slab 加密度硬遮罩，也没有约束分量永远留在初始 slab；高斯尾部可以跨边界贡献。相邻 slab 之间目前也没有进一步共享二维字典。

## 目前结果应怎样理解

在一个 64×64×30 的原生子体上，使用 64 根柱、每柱 4 个分量、16 个训练视角和 200 步更新：

| 初始化 slab 层数 | 最终训练视角渲染 PSNR | 最终原生 30 层密度 PSNR |
|---|---:|---:|
| 8 | 31.40 dB | 15.73 dB |
| 16 | 31.86 dB | 15.96 dB |

这是两种不同指标：前者比较透射图像，后者比较体素位置处的密度值。补零后的全体密度 PSNR 又会包含大量背景，因此也不能直接替代原生 30 层指标。

模型训练参数量为 `M * (5 + 3*K)`。本例是 `64 * (5 + 3*4) = 1088` 个 float32 标量，即 4352 字节。这不包含 Adam 状态、NPZ 元数据、场景展开数据及导出文件开销。

这些结果验证了共享训练和恢复流程可以工作。当前尚未进行等参数量或等质量的独立 3D Gaussian 对照，没有测得原生柱求交加速，也没有实现 Gabor 柱训练。下一步应把“表示拟合质量对照”和“同一份参数下的求交效率对照”分别验证。

完整原始结果见 [实验目录说明](results/gaussian_column_training_results/README.md)；建议优先看 `slab08_voxel_bound` 和 `slab16_voxel_bound`，不要把早期 `slab08` 目录混入最终比较。
