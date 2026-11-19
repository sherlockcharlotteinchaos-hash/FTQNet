# FTQNet - MicroSTNet Author-Code Protocol Alignment

This source release makes *only* M3-stool, A-stool, A-saliva and F4-stool follow the
public MicroSTNet preprocessing / evaluation protocol. The existing FTQNet
model (including FLinear), existing non-microbial datasets, their scripts,
and their training / result output formats are unchanged.

## Download datasets yourself

Put the untouched CSVs **directly** in `dataset/` (no new subdirectory):

| `--data` | Original filename |
|---|---|
| M3-stool | `M3_gut_abun.csv` |
| A-stool | `A_Stool_abun.csv` |
| A-saliva | `A_Saliva_abun.csv` |
| F4-stool | `F4_gut_abun.csv` |

The ZIP intentionally includes none of these CSVs, no weights and no results.
There is no on-disk data conversion, cache or extra diagnostic output.

## Exact protocol taken from the public author implementation

References:
- Paper: https://doi.org/10.1099/mgen.0.001519
- Data prep: https://github.com/pwner-web/MicroBioSTNet/blob/main/lib/prepare.py
- Training: https://github.com/pwner-web/MicroBioSTNet/blob/main/main.py

1. Read input CSV exactly as a taxon x sample matrix. Preserve original rows
   and columns. No genus collapsing, train-only OTU filtering, resampling,
   depth normalization, square-root transform or extra cleaning. The author's
   dataset is not ETT formatted; adapter transposes in memory.
2. Compute one global mean and standard deviation across *all* taxon / sample
   entries (float32), Z-score the whole matrix, add +1 to both inputs and
   targets. This **uses validation-period statistics**, exactly as `get_()`
   does, and is not leak-free. Negative/non-finite entries are rejected; zeros are preserved;
   all-zero taxa were absent in the two supplied CSVs examined here.
3. Split the original sample order at `int(0.7 * num_samples)`, without
   shuffling sample dates. Window each partition independently: input 12,
   output 1 or 8; validation inputs **do not** use training-period history.
4. Reproduce the published **non-LSTM** training-data overlap: `cut_datasets`
   adds `int(0.1 * len(validation_windows))` regularly spaced validation
   windows, plus the final validation window if not already included, to the
   train windows while retaining ALL validation windows for model selection.
   In the paper code this is enabled with `--ratio 0.1`.
5. Adam, `learning_rate=0.001` constant, loss `L1Loss` (MAE), batch size 8,
   200 full epochs with no early stopping, random seed 7. Select weights at
   the *lowest full-validation MAE*, not best MSE. No OneCycle schedule and
   no per-batch-unweighted validation average. `--patience 201` is present
   only for script completeness; this isolated trainer runs the full budget.
6. The `test` mode intentionally uses the SAME 30% validation windows and
   outputs validation MAE/MSE in FTQNet's existing `result.txt` and
   `results/<dataset>.txt` format. There is **no independent 20% test** in
   the author's 70/30 implementation. Do not call this held-out test MAE.

**Two discrepancies in the original paper:** its methods text also describes
random 80/20 and its subject-A sample count differs from the released CSV
provided here. This implementation prioritizes the exact public Python
`get_()` / `main.py` settings over contradictory prose. For example, the
uploaded A-stool CSV has 341 sample columns, not the 299 stool samples given
in the methods narrative. Consequently no exact reproduction of Table 1 is
promised. Published MicroSTNet numbers also use another model architecture.

**Not architecture-matched:** FTQNet remains FTQNet. Its `cycle=1`,
`use_revin=0`, `d_model=64`, `dropout=0`, attention setup, `FLinear(alpha=0.9,
c=1, groups=8)` and optimizer internals are fixed, listed explicitly in
scripts, and are not parameters of the MicroSTNet graph model. A fair *model*
comparison requires both models to run on an identical matrix of examples,
identical scaling and identical evaluation definition; the published best
validation MAE is only a reference until this is independently verified.

## Run on Linux / WSL / Git Bash

```bash
conda env create -f environment-epicast.yml
conda activate ftqnet-epicast

# Each script lists two fully separate commands and runs both by default.
bash scripts/TQNet/m3_stool.sh
bash scripts/TQNet/a_stool.sh
bash scripts/TQNet/a_saliva.sh
bash scripts/TQNet/f4_stool.sh

# Or select a single horizon:
bash scripts/TQNet/a_stool.sh 1
bash scripts/TQNet/a_stool.sh 8
```

Data path and filenames are unchanged. `--enc_in 0` is automatically resolved
from the **original number of rows** (not filtered taxa), without adding new
CLI arguments. No fabricated date features are used, and H=8 means eight
sample positions, not necessarily eight calendar days.

Only these existing result paths are written:

```
result.txt
results/<dataset>.txt
checkpoints/<setting>/checkpoint.pth
test_results/<result_setting>/*.pdf
```

A-saliva's source CSV was not supplied during this edit and must be provided
before its experiments can run. Other datasets follow their original paths.

---

## Existing original tasks (preserved)

# FTQNet / FLinear：EpiCastBench 周数据集

COVID-19 与 ILI 共用 `exp/exp_health.py`、`data_provider/data_health.py`。COVID-19 按原脚本固定采用 `--covid_calibrated` 路径：先训练，再仅用验证集选择并冻结校准，最后测试。校准、约束与评价逻辑已并入 `exp_health.py`，分组识别和训练集零点归一化并入 `data_health.py`；原两个独立校准文件已删除。保留现有脚本参数及 ILI/Wearable 的共用普通评价流程。

五个周数据集的读取代码统一放在 `data_provider/data_epicast.py`，实验代码统一放在 `exp/exp_epicast.py`。`run.py` 和 `data_factory.py` 已统一接入新模块；各数据集仍保留原有的专用处理、模型、损失、时间划分和输出格式。此次仅整合文件与入口，不调参、不修改共享模型或实验脚本。

本工程接入 Influenza US，并保留 Chickenpox Hungary、Dengue Colombia、Dengue Argentina、Influenza US 2 的既有数值流程。共享 `models/FTQNet.py`、`layers/FLinear.py`、其他数据读取器和其他实验实现保持上传版原样。固定起点实验引擎仅增加工厂接口，原四个任务的默认模型、数据、训练、校准、指标和输出逻辑不变；没有旧版回退或额外命令行参数。

## 环境

```bash
conda env create -f environment-epicast.yml
conda activate ftqnet-epicast
```

目标：Python 3.8.20、PyTorch 2.4.1+cu121、CUDA Toolkit 12.1、NumPy 1.24.4、Pandas 2.0.3，依赖见 `requirements-epicast.txt`。本次实际验证环境为 Python 3.12.14、PyTorch 2.5.1+cpu、NumPy 2.3.5、Pandas 2.2.3；当前没有 CUDA GPU，也没有收到环境截图，不能声称已在目标 GPU 环境实测。

## 数据

自行解压原始 CSV，直接放入 `dataset/`。不要改名或建立数据集子目录。程序不解压、不下载、不修改、不另存数据。

| --data | 原始文件名 | 重建网格 |
|---|---|---|
| Influenza-US | Influenza US.csv | 521 × 53 |
| Influenza-US-2 | Influenza US 2.csv | 1200 × 10 |
| Chickenpox-Hungary | hungary_chickenpox.csv | 522 × 20 |
| Dengue-Colombia | Dengue Colombia.csv | 835 × 33 |
| Dengue-Argentina | Dengue Argentina.csv | 92 × 23 |

Influenza US 是 CDC 州级长表，不能把原始 CSV 的行数当成时间长度，也不能混用 HHS 十地区 US2。按 MMWR 年/周生成 Sunday 起始日期，仅保留 2010-10-03 至 2020-09-20 的 `States / ILITOTAL` 人数，不使用 ILI 百分比作为标签。排除全缺失的北马里亚纳群岛和单列的 New York City，保留 50 州、DC、Puerto Rico、Virgin Islands，共 53 个地区，按地区名排序。重复周/地区、非法周数、负值、无穷值和维度错误直接报错。

此地区选择加首报前零初始化，得到与论文相符的 521 周、53 地区和约 4.20% 零值比例，但这仍是基于原始导出的重建推断，未与作者整理版逐元素核对。Puerto Rico 和 Virgin Islands 在开始部分缺少记录；LOCF 无法填充没有过去观测的首报前缺口，这些位置明确以零初始化，既不倒填也不使用未来值。

病例人数原始网格（缺口 LOCF、首报前零初始化）用于标签、指标和 MASE 分母。零机构、零就诊且零病例标记为无上报：只对模型输入因果前填，训练损失屏蔽无上报标签，保留有监测时的真实零病例；不删极端疫情、不平滑或改写测试标签。单个地区缺报不会导致其余 52 个地区的同周训练标签被丢弃。最终验证和测试要求全部目标已上报。

原四个任务的处理保持上传版：水痘按原 Date 读取；登革热按国家筛选 Admin2 / Week / Total 后汇总省级、因果 LOCF；US2 读取十个 HHS 地区并保留其原有缺报训练窗口筛选。PAHO 市县缺报完整性仍有不确定性，不能宣称与论文整理版完全相同。

## 运行

```bash
# 五个任务各 4/8/12 周，共 15 组独立训练和测试
bash run_epicast_fixed.sh

# 每个数据集一个脚本；每个 H 独立列出完整命令
bash scripts/TQNet/influenza_us.sh
bash scripts/TQNet/influenza_us_2.sh
bash scripts/TQNet/chickenpox_hungary.sh
bash scripts/TQNet/dengue_colombia.sh
bash scripts/TQNet/dengue_argentina.sh

# 可仅选部分 H
bash scripts/TQNet/influenza_us.sh 4 8

# CPU 运行
CUDA_VISIBLE_DEVICES="" bash scripts/TQNet/influenza_us.sh
```

脚本显式列出已有的 TQNet、FLinear、数据、训练和设备参数。共享模型 attention 固定为 4 heads、dropout=0.5，无对应可配置 CLI，脚本有注释；US2 的原 12 周实例仍使用原有 attention dropout=0。AMP、多 GPU、FLOP action flags 保持默认关闭。没有新增参数。

仅测试已保存的模型：复制对应 H 的完整训练命令，将 `--is_training 1` 改为 `--is_training 0`，其他参数保持一致。原始 CSV、协议、地区顺序、预测长度和推断参数必须与 checkpoint 匹配。

## 时间协议和训练

每个 H 独立训练，输入固定为 24 周。训练样本按周滑动，输入和完整 H 周标签均不得越过所属训练边界。

最终唯一预测起点 `O=T-H`；输入为 `[O-24,O)`，一次性预测 `[O,T)`，不滚动测试，不用真实测试值更新输入。Influenza US 三个起点分别为 2020-08-30、2020-08-02、2020-07-05。

Influenza US 在 O 之前留出 12 周历史验证段，验证预测标签均不超过 O；4/8/12 周分别有 9/5/1 个完整验证窗口。H=12 仅一个历史验证起点，模型选择方差可能较大。此近期划分让当时已知的 2020 年早期监测冲击进入训练历史，避免用全在冲击前的训练段选择夏季预测。原四个任务的划分不变。

先在验证段之前的训练历史拟合统计量、季节初始化和模型，按历史验证 MASE 选择训练轮数与可选近期微调。总预算由现有 train_epochs 分给全历史训练（80%）和近期微调（20%），patience 控制提前停止，不强制训练满预算。冻结选择后重新设定种子，在全部 `[0,O)` 历史重新拟合统计量并按所选轮数重训。最终测试标签不参与选择。

Influenza US 复用完整 FTQNet 与 FLinear，不加可训练参数或外部权重：

- 按训练前缀的已上报历史初始化已有 seasonal temporalQuery，并对齐共享模型查询的起点相位。
- 4 周使用局部原始人数相对坐标，末 8 周平均活动水平决定残差尺度；8/12 周使用局部 log1p 坐标，24 周统计用于输入归一化，末 8 周波动决定残差尺度。
- 4 周以输入末 4 周算术平均为锚点，避免单周报告突增直接成为整个预测块的水平；8 周以最近已知病例数为锚点；12 周使用输入末 4 周的衰减趋势锚点。
- 使用逐地区 MASE 训练损失，屏蔽无上报标签，并沿用 US2 的输入活动水平平衡思想，避免历史冬季高峰完全支配低活动期拟合。
- 8/12 周的历史验证和最终推断共同使用原有 `flu_constraints` 因果函数，包括近期波动带、下降约束及峰后冲击收缩；不进入训练损失。4 周只做非负截断，不向可能孤立突增的末值强制收缩。所有阈值来自固定规则或输入历史，不读取未来病例/机构数。

因此完整方法包含预处理、锚点和推断约束，不能把全部效果只归因于 FTQNet/FLinear。

## 指标与输出

MASE 对每个地区计算 `mean(abs(pred-truth)) / mean(abs(diff(history)))`，分母只用 `[0,O)` 的原始重建病例，最后对 53 个地区等权平均。训练活动权重不用于评价。MAE、RMSE、SMAPE 在原始病例尺度逐地区计算再平均；MSE/MAE 的标准化口径沿用原项目。

只生成既有类型和路径的实验文件：

```text
checkpoints/<setting>/checkpoint.pth
results/<dataset>.txt
result.txt
test_results/<result_setting>/0.pdf
```

保留上传版实际的 result.txt 行格式（setting；mse/mae/mase/MAE），PDF 目录沿用省略 FLinear 参数段的命名。原始尺度四项指标写入既有日志。不生成清洗 CSV、预测 NPY、额外指标表、诊断文件或预训练权重文件。

checkpoint 内保存模型、训练前缀统计、选择结果、地区、起点、原始文件 SHA-256 和推断参数。训练和重载输出可直接核对。

源码包不含数据、结果、日志、checkpoint、缓存或开发验证程序。实验结果另包提供。

## 论文对照边界

论文 https://arxiv.org/html/2605.11598v1 ，附录 Tables 2/6/10 分别给出周数据 4/8/12 周的 MASE，应与每个数据集每个 H 的全部方法最小值对照。

论文描述 Darts、24 点输入、12 点训练预测块、50 epochs、seed 42，同时使用 fixed-origin 和 rolling-window 两种表述。本工程优先遵从用户要求：每个 H 独立训练、从 T-H 只预测一次，因此训练块 H=4/8 时与论文描述不同；整理数据未逐值对齐，不能宣称严格同协议复现或独立盲测 SOTA。上传版开发已查看历史末尾测试结果，本轮也进行过探索性尾段试验；最终测试数值仅作论文参考比较。原任务结果变化还可能来自运行环境差异，默认计算逻辑没有改动。

## F4-stool integration (same author-code protocol)

The separate `scripts/TQNet/f4_stool.sh` runs 12-to-1 and 12-to-8 with full FTQNet/TQNet/FLinear flags, 200 epochs each, and the same validation MAE checkpoint rule. Original `F4_gut_abun.csv` contains 225 taxon rows and 130 sample-order columns. The loader does not filter, transpose on disk, or create converted datasets. Results continue to go to the original `result.txt`, `results/F4-stool.txt`, `checkpoints/`, and `test_results/` locations.
