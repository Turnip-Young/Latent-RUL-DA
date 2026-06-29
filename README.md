# 基于域自适应 Transformer 的跨故障剩余寿命预测

本项目围绕工业设备剩余寿命预测（Remaining Useful Life, RUL）中的跨故障迁移问题展开，基于论文 **Domain Adaptive Remaining Useful Life Prediction With Transformer** 进行复现与改进。原模型主要通过特征级对齐和语义级对齐缓解源域与目标域分布差异；本项目进一步针对单故障域到多故障域迁移时容易出现的负迁移问题，引入潜空间退化特征约束，使模型更关注与寿命衰退相关的时序变化。

项目使用 NASA C-MAPSS 数据集，重点实验任务包括：

- `FD001 -> FD003`
- `FD002 -> FD004`

其中 `FD001`、`FD002`、`FD003`、`FD004` 表示不同故障模式和不同工况下的发动机退化数据。

## 项目改进

原始复现模型对应代码中的 `--type 2`，即 Transformer backbone + 输出层对抗对齐。该方法在一般域自适应任务中有效，但在单故障到多故障迁移场景下，直接强制源域和目标域分布对齐可能会把目标域中真实存在的多故障退化结构压平，从而带来负迁移。

本项目改进模型对应 `--type 3`，主要做法如下：

- 引入潜空间退化表征，将潜变量拆分为退化相关因子和故障相关因子。
- RUL 预测主要依赖退化因子，减少故障模式差异对寿命预测路径的干扰。
- 使用重构约束、平滑约束、去相关约束、单调性约束和退化排序约束，增强潜空间中退化特征的可解释性和连续性。
- 不再直接对源域和目标域进行强对齐，而是通过退化先验约束引导模型学习更稳定的跨故障退化表示。

简而言之，本项目不是继续增强源域和目标域的硬对齐，而是让模型先学到更像“退化过程”的表示，再用于跨故障 RUL 预测。

## 环境配置

建议使用原论文一致或相近的 PyTorch 环境：

```bash
python >= 3.8
pytorch == 1.10
torchvision == 0.11.0
numpy
pandas
scikit-learn
```

如果使用 Conda，可以参考：

```bash
conda create -n rul_transformer python=3.8
conda activate rul_transformer
pip install torch==1.10.0 torchvision==0.11.0
pip install numpy pandas scikit-learn tqdm
```

## 数据准备

项目已经按照代码需要组织 C-MAPSS 数据，主要目录如下：

```text
CMAPSS/        # C-MAPSS 数据文件
save/          # 数据划分文件与保存的模型权重
logs/          # 训练和验证日志
online/        # 训练过程中保存的在线权重
```

如果重新配置数据，需要将 C-MAPSS 数据放在 `CMAPSS/` 目录下，并保证 `save/FD001`、`save/FD002`、`save/FD003`、`save/FD004` 等数据划分文件存在。

## 模型类型

训练脚本通过 `--type` 选择不同方法：

| 参数 | 方法含义 |
| --- | --- |
| `--type 0` | 仅输出层对抗对齐 |
| `--type 1` | DANN backbone 特征对齐 |
| `--type 2` | 原始复现模型：backbone + output 对抗对齐 |
| `--type 3` | 本项目改进模型：潜空间退化特征约束，不做源域-目标域硬对齐 |

本项目主要使用 `--type 3`。

## 训练方法

### 训练 FD001 -> FD003

```bash
python train_cmapss.py --source FD001 --target FD003 --type 3 --gpu 0 --run_id FD001_to_FD003_latent
```

训练完成后，模型权重会保存到：

```text
save/final/latent_13.pth
online/13_latent_net.pth
```

### 训练 FD002 -> FD004

```bash
python train_cmapss.py --source FD002 --target FD004 --type 3 --gpu 0 --run_id FD002_to_FD004_latent
```

训练完成后，模型权重会保存到：

```text
save/final/latent_24.pth
online/24_latent_net.pth
```

## 验证方法

### 验证 FD001 -> FD003

```bash
python validation_cmapss.py --source FD001 --target FD003 --type 3 --model_path save/final/latent_13.pth --run_id eval_latent_13
```

### 验证 FD002 -> FD004

```bash
python validation_cmapss.py --source FD002 --target FD004 --type 3 --model_path save/final/latent_24.pth --run_id eval_latent_24
```

验证结果会保存到 `logs/` 目录下，包括：

```text
eval.log
config.json
eval_units.csv
summary.json
```

其中 `summary.json` 记录平均 RMSE、Score 和测试样本数量。

## 原模型对比运行

如果需要复现原始对抗对齐模型，可以使用 `--type 2`。

训练原模型：

```bash
python train_cmapss.py --source FD001 --target FD003 --type 2 --gpu 0 --run_id baseline_13
python train_cmapss.py --source FD002 --target FD004 --type 2 --gpu 0 --run_id baseline_24
```

验证仓库中已有的原模型权重：

```bash
python validation_cmapss.py --source FD001 --target FD003 --type 2 --model_path save/final/both_13.pth --run_id eval_baseline_13
python validation_cmapss.py --source FD002 --target FD004 --type 2 --model_path save/final/both_24.pth --run_id eval_baseline_24
```

如果使用自己重新训练得到的 `type 2` 权重，可以将 `--model_path` 改成对应的权重路径。

## 实验结果

评价指标中 RMSE 越低表示 RUL 预测误差越小。根据当前实验日志，本项目改进后的 `type 3` 模型在 `FD001 -> FD003` 和 `FD002 -> FD004` 两个跨故障迁移任务上均优于原始 `type 2` 对抗对齐模型，说明潜空间退化特征约束能够降低跨故障迁移中的负迁移影响。

| 迁移任务 | 模型 | 测试样本数 | RMSE | Score | 结果说明 |
| --- | --- | ---: | ---: | ---: | --- |
| `FD001 -> FD003` | 改进模型 `type 3` | 100 | 14.7310 | 3965 | RMSE 低于原始 `type 2` 模型 |
| `FD002 -> FD004` | 改进模型 `type 3` | 249 | 12.0304 | 881 | RMSE 低于原始 `type 2` 模型 |

对应日志文件：

```text
logs/FD0013/evalue/eval_FD001_to_FD003_type3_20260616-125739/summary.json
logs/FD0024/evalue/eval_FD002_to_FD004_type3_20260615-124510/summary.json
```

这两个结果说明，在单故障域迁移到多故障域的场景下，直接进行源域-目标域强对齐并不一定最优；通过退化潜空间建模，让模型显式关注退化趋势和寿命相关特征，可以取得更稳定的跨故障 RUL 预测效果。

## 主要文件

```text
train_cmapss.py        # 训练入口
validation_cmapss.py   # 验证入口
model.py              # Transformer RUL 模型与潜空间模块
loss.py               # 对抗损失、潜空间约束和退化先验损失
dataset.py            # C-MAPSS 数据读取与窗口构造
logs/                 # 实验日志
save/final/           # 最终模型权重
```

## 参考论文

```bibtex
@article{li2022domain,
  title={Domain Adaptive Remaining Useful Life Prediction With Transformer},
  author={Li, Xinyao and Li, Jingjing and Zuo, Lin and Zhu, Lei and Shen, Heng Tao},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={71},
  pages={1--13},
  year={2022},
  publisher={IEEE}
}
```
