# Local Axial SVD Analysis

`local_svd_analysis.py` 使用 PyTorch 对原生 PNG 切片进行局部分块分析。
原有 `svd_analysis.py` 保留，仍可运行全图 SVD。

相关脚本、测试和已有实验结果统一位于 `D:\gabor_fields\svd`。
默认输出位于该目录下的 `local_svd_results`；原始 PNG 仍从 `--input` 指定目录读取。
已有结果中的配置文件记录了运行时的历史路径，移动目录不会改变其数值结果。

---

## Install

复用主项目根目录的 `.venv`，依赖统一声明在根目录 `setup.py` 的
`svd` extra 中。环境安装与 CUDA 版本选择见[主项目 Install](../README.md#install)。
以下命令均从 `gabor_fields` 根目录执行；本目录不创建单独的 Python 环境。

```powershell
uv pip install --python .venv/Scripts/python.exe -e ".[svd]"
.\.venv\Scripts\Activate.ps1
```

Linux/macOS 使用 `.venv/bin/python`，激活命令为 `source .venv/bin/activate`。

---

## Analyze

```powershell
python svd/local_svd_analysis.py --input D:\SVD
```

默认扫描 resolution=128/64/32/16 × slab=4/8/16/30，共16组。
参数均可指定单个或多个值：

```powershell
python svd/local_svd_analysis.py --input D:\SVD --resolution 64 --slab 8 --device cpu
python svd/local_svd_analysis.py --input D:\SVD --resolution 64 32 --slab 8 16 --batch-size 128 --no-plots
python svd/local_svd_analysis.py --help
```

全图 SVD 对照：

```powershell
python svd/svd_analysis.py --input D:\SVD --output svd/global_svd_results
```

- `--resolution`：原始像素上的横向分块边长，不缩放图像。
- `--slab`：轴向每段最多包含的切片数，按固定步长分割。
- `--device auto/cpu/cuda`：auto 实测 CUDA 内核可用性，不兼容时警告并回退 CPU。
- `--dtype float64/float32`：默认 float64，控制 Gram 累加与特征值计算精度。
- `--batch-size`：每批同形状块的数量，默认128。显存不足时减小。
- `--ranks`：指定保留模式数，自动加入0与该配置最大秩作为参考。
- `--energy-epsilon`：近零块的均方能量阈值，默认1e-12，只影响分类和有效秩，不删除重建误差。
- `--examples`：每种分析的示例块数量，默认2。
- `--no-plots`：仅保存数值结果。

---

## Method

图片按自然数字顺序读取，转换灰度并除以255，堆叠顺序为[z,y,x]。
横向与轴向均不重叠、不补零，边缘块保留真实尺寸。
例如30层按slab=8分为[0,8)、[8,16)、[16,24)、[24,30)。

每块整理为[height*width,depth]矩阵。raw分析原始值；axial_centered减去每个像素在当前slab内的均值。
常规批处理只计算奇异值平方，rank-r误差等于未保留谱的总和。
去均值重建须加回均值，其参数容量也计入均值图。

整体MSE是所有块SSE总和除以总体素数，不直接平均块PSNR。
PSNR峰值固定为1。CSV中的inf表示零误差；严格JSON将非有限数存为null。
各配置报告all、full_blocks、signal_blocks三个范围；signal_blocks指当前分析中非近零的块。
因此centered的signal_blocks表示存在轴向变化的块，与raw的信号块不是相同筛选。

容量以浮点因子标量数r*(pixels+depth+1)估算；centered额外加pixels。
scalar_compression_ratio是原始标量数/因子标量数，不是8-bit PNG到浮点因子的字节压缩比。
这是自由二维模板的SVD容量参考，不是Gaussian/Gabor模型大小。
完整秩能够恢复原数据是数值自检，不作为低秩共享证据。

---

## Output

默认输出 -> `svd/local_svd_results/`。所有名称以 `results` 结尾的目录
均由主目录 `.gitignore` 排除，不提交实验输出。

每组保存到resolution_064/slab_08等目录，包含config.json与raw/axial_centered：

- summary.json：范围统计、有效秩、误差与容量。
- rank_metrics.csv：固定rank指标。
- block_metrics.npz：逐块bounds、谱、能量、有效秩、误差等数组。
- 秩分布、空间秩图、空间误差图与少量重建示例。

根目录保存experiment_config.json、slice_order.txt、comparison.csv和跨配置总览图。
bounds数组顺序为[z0,y0,x0,z1,y1,x1]，结束索引不包含在块内。
运行中每完成一组便保存comparison.csv，长扫描中途退出后已完成结果仍可读取。

---

## Tests

```powershell
python -m unittest discover -s svd/tests -p test_local_svd.py -v
```

验证覆盖尾块、零输入、轴向常量、已知rank1、NumPy直接SVD对照、谱尾误差、均值恢复、批大小一致性及可用时的CUDA一致性。
CUDA计时包含同步；metadata记录实际设备、初始传输与核心分析时间，绘图和PNG加载不包含在核心时间中。
