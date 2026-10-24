# Q1：Windows／WSL 的 RTX 4070 8GB 运行说明

2026-10-24 更新：本轮显存检查和 4800 次完整训练已经在用户的 RTX 4070 上完成并返回；下文保留可复现的操作步骤。结果核验、实际耗时和未达到的预测目标见[本轮结果](validation/q1-modular-64-wsl-2026-10-24/README.md)。

用户在 2026-10-23 提供的终端输出确认：WSL 可以识别 RTX 4070 Laptop GPU，驱动 580.88；独立环境 Python 3.12.15、NumPy 2.5.3、PyTorch 2.14.0+cu126 已完成小矩阵 CUDA 前向／反向。这是用户提供的环境证据，不是完整模型、长期积分或显存验收。

本任务沿用已生成的 64 基因公共 RNA，不重新生成数据，不使用生成器、真实 GRN、G/U 状态或效率。六个观测时间（含真实长期时间 64）、32／16 靶点划分、模型结构、64 细胞／条件、每次参数更新 4 条件、4800 次更新均不改变。

## 降低显存但保留训练目标

`--condition-microbatch 1` 将一次有效更新的 4 个条件逐个积分和反传，按条件数加权累积梯度，最后只裁剪、更新一次。它保留同一批数据的平均损失；不删细胞或时间，不是把有效批量改成 1。不同矩阵批形状和求和顺序仍可能造成浮点差异，不保证长期优化逐位相同。

`--checkpoint-every 60` 定期保存完整训练状态。只在一次完整优化更新后保存；普通进程中断可回到最近成功保存的区间，模型、优化器及随机数状态一同恢复。磁盘故障或突然断电仍可能损坏文件，不能承诺绝对数据安全。恢复仍使用原先的总更新预算，不追加 4800 次。CPU 单元测试能验证恢复流程，不保证跨 GPU、驱动或软件版本逐位一致。

当前最终模型保存后，评分与步长减半检查仍在 CPU 运行。这一阶段可能继续耗时；GPU 利用率下降不等于整个任务已结束，以最后的报告和退出状态为准。

## 文件与位置

交付一个 `grn-q1-wsl-20261023.tar.gz` 代码＋公共数据归档和独立 SHA-256 值。包含白名单学习器源码、启动／显存检查脚本、单元测试及 `data/q1-modular-64/`。不包含 Docker 镜像、虚拟环境、Git 历史、生成器或 `private/`。原 H200 镜像和公共数据归档不覆盖、不重新贴同一镜像版本标签。

把归档放在 Windows 用户 Research 的“下载”目录；如果实际 Windows 用户目录不同，调整下面 `/mnt/c/Users/Research/Downloads/`。在已激活 `(grn-q1)` 的 Ubuntu 终端执行：

```bash
cd /home/research
tar -xzkf /mnt/c/Users/Research/Downloads/grn-q1-wsl-20261023.tar.gz
cd /home/research/grn-q1-wsl-20261023
sha256sum -c SHA256SUMS.txt
```

`-k` 表示拒绝覆盖已存在的文件。如果同名目录已经存在，先停止确认，不删除旧结果或覆盖环境。不使用 Windows 系统目录运行项目；输入和结果放在 WSL 的 Linux 用户目录中。

## 第一步：真实最长时间显存检查

```bash
bash tools/scmultisim/run_q1_wsl.sh probe
```

只读取共同基线和预先登记的前四个训练靶点在真实长期时间的 RNA，实际完成一个有效 4 条件更新；每个条件 64 细胞，模型和积分步长与完整训练一致。不读取测试靶点响应、不计算测试分数、不使用第三重复拟合。保存实际耗时、CUDA 峰值显存、环境与文件校验。报告失败不能当作通过；显存不足时停止，不自动减数据或改模型。

此检查发生在完整训练以外；所得权重被丢弃，不用来初始化或挑选最终模型。通过只支持这个初始模型、批量和长期样本能运行，不能保证后续 4800 次更新不会出现非有限损失，也不代表模型预测有效。

先把 `results/q1-modular-64-wsl-seed20261022/gpu_probe.json` 及日志发回审阅，再启动完整训练。

## 第二步：完整训练

显存检查通过且确认后：

```bash
bash tools/scmultisim/run_q1_wsl.sh train
```

笔记本插电、保持散热、不要睡眠或关闭 WSL。保留原始测试划分，不根据测试分数挑选更新次数。入口会核对 probe 使用的数据、源码、软件版本和显卡名称；不同则拒绝。已有训练目录不会被新任务覆盖。

中断后先确认旧进程已经退出，再在同一环境、同一目录执行：

```bash
bash tools/scmultisim/run_q1_wsl.sh resume
```

只有已经保存了完整训练检查点才能恢复；没有检查点不应伪称续跑。代码、数据、配置或设备改变时拒绝续跑。已完成的结果不能再用此命令追加训练。检查点使用受限的 PyTorch 加载接口，但仍只加载自己产生且可信的文件，不接受来源不明的检查点。

## 输出与回传

**唯一需要回传的目录：**

`/home/research/grn-q1-wsl-20261023/results/q1-modular-64-wsl-seed20261022/`

内含 `run.log`、`gpu_probe.json`；完整训练后内部 `results/` 包含 `run_plan.json`、`training_checkpoint.pt`、`progress.jsonl`、最终 `model.pt`、`report.json`、分数、预测和源码快照。检查点不等于最终模型，最终模型也不等于训练效果通过验收。

只打包这个运行目录，不压缩项目、输入数据或环境：

```bash
tar -C /home/research/grn-q1-wsl-20261023/results \
  -czf /home/research/q1-modular-64-wsl-results.tar.gz \
  q1-modular-64-wsl-seed20261022
sha256sum /home/research/q1-modular-64-wsl-results.tar.gz
cd /home/research
explorer.exe .
```

即使失败也保留该目录和日志。这里只准备单网络开发训练，不宣称完成正式盲测或真实 GRN 恢复。4070 的实际显存和完整训练时间必须由这台机器运行后报告，不能从 H200 或 Mac 小检查推断。
