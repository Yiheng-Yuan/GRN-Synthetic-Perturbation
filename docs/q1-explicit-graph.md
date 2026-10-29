# 显式 GRN 与 RNA 单状态开发训练

2026-10-29。复用已有64基因数据，不重新生成。本入口补齐最初要求的图结构，不把旧黑盒预测运行改称已完成结构恢复。软件测试通过；用户回传新模型GPU探测通过，首组lambda_W=0到120/4800次更新，尚未完成完整训练或独立结果验收。此进度仅据用户终端日志，不冒充已下载复算的结果。[进度记录](validation/q1-explicit-graph-wsl-2026-10-29.json)。旧镜像`grn-q1-rna:20261024-dev1`不包含本模型。

## 1. 实际模型

`src/grn_experiment/q1_graph.py::GraphRNAODE`只传播成熟RNA状态x，没有隐藏记忆h，也不输入真实网络、效率、未剪接RNA或真实速度。显式RNA历史比较留待下一步。

\[
Z=W\odot H,\qquad H=2\tanh(H_{\rm raw}),
\]
\[
\dot x_i=(1-\rho_i u_i)c_i\operatorname{softplus}
\left(b_i+\psi_i\left[\sum_{j\ne i}Z_{ij}\,
\phi_j\left(\log(1+x_j)/s_j\right)\right]\right)
-\operatorname{softplus}(\delta_i)x_i.
\]

- 行i接收、列j发出：`W[i,j]=1`表示j→i。对角线固定为0，不把自身消耗项当调控自环；所有非对角方向均可候选，不输入真图掩码或真实边数。
- phi和psi是可学习的单调神经函数。源函数在0处为0、在1处为1；接收函数在0处为0。H的符号因此与非零直接产生作用的符号一致。H有界与归一化只能减少尺度自由，不证明边强度唯一可辨识。
- s与c仅由前两个共同基线重复初始化／固定；s=max(mean(log1p(x)),1)，c=0.1×max(mean(x),1)/log(2)。b、delta、非线性网络、H与门控参数可学习。
- rho=0.5+0.5sigmoid(raw_rho)，同一基因跨细胞、时间、强度共享。未受训练扰动的靶点仍为初始0.75，明确标记未估计。
- 单调性、加和图输入、非负产生及线性降解是公开建模先验。它不是无限制函数发现，也不是模拟器精确的隐藏G/U/M方程；仅观测M时使用有效闭合近似。
- 训练使用hard-concrete可微松弛，温度2/3、拉伸区间[-0.1,1.1]。解析`P(gate>0)`之和为期望边数，和裸sigmoid或H的L1不同。一次微批次的处理与对照共用一张门控图，整段积分不换图。
- 验证、最终预测和导出统一为`W=1[P(gate>0)>=0.5]`，不以真实A选阈值。松弛训练与硬图预测可能有差距，以实际硬图验证成绩暴露该问题。
- 使用最大步长0.25的可微投影RK4推进；截断会改变边界附近的形式精度。每次完成训练后，对固定训练靶点的最长时间用0.25/0.125比较，8个预测细胞，最大基线标准差缩放差须≤0.01。该检查失败则停止完成登记；它不是稳态或传播正确性的证明。

## 2. 损失、划分与选择

`src/grn_experiment/q1_graph_train.py`实现：

\[
L=L_{\rm mean}+0.1L_{\rm population}
 +\lambda_W\sum_{i\ne j}P(W_{ij}=1).
\]

均值项使用同条件全部拟合细胞的逐基因均值，处理／对照各半；除以训练共同基线的逐基因标准差（下限1），再平方并按条件与基因平均。各采样时间等频率轮转。分布项以`log1p(RNA)/s`比较不配对群体，逐基因平方Wasserstein距离与联合投影距离各半；24个联合方向每轮刷新，两臂各半。不设置独立响应差损失。稀疏罚项每次参数更新只计一次，不随条件微批次数倍增。

固定数据时点0.25、1、2、4、8、64，加一次共同t=0基线。初态从前两个基线重复抽样；拟合只用前两个处理／对照重复。内部验证使用其靶点的第三重复，所有时间随靶点一起分组。不能按行拼接为同细胞轨迹。

1. 仅解码原32个训练靶点，从中按种子20261022划24拟合／8内部验证；原16留出靶点的响应文件此时完全不读取。
2. 依次比较lambda_W={0,1e-5,1e-4,1e-3}，每组4800次更新、每次4条件×64细胞、微批次1、学习率0.001；hidden=8是当步神经特征数，不是传播记忆维度。
3. 用内部验证绝对表达误差选系数。绝对差≤1e-6时优先较少二值边，再较小lambda；不使用稀疏加权训练总损失或真实图选择。
4. 按选中设置用全部32训练靶点重拟合4800次，先保存模型、图和预测锁，再读取原16靶点做开发评分；同时报告无响应与直接迁移基线。原16靶点曾查看，不能称新的正式盲测。
5. 无分布项消融和显式RNA历史图模型未加入本轮自动任务，待这轮结果后单独运行；本轮也不自动启动正式20网络实验。

每轮日志保存未加权误差、加权贡献、预期边数、硬图边数、各参数梯度范数和实际抽到的靶点。权重不是可训练参数。文献依据见[四篇核查](protocol/q1-loss-literature.md)。

## 3. 在 Windows 的 Ubuntu 执行

本轮使用Ubuntu已有`/home/research/.venvs/grn-q1` Python虚拟环境，不是旧Docker容器；没有新建环境或安装／升级依赖，EEG环境不改动。代码及新增结果位于D盘，既有合成数据继续从原路径读取。原始训练交付源码为提交`990f8d8`，后续本轮main更新仅增加文档、进度与数据发布记录，不改变正在运行的训练源码。

Windows端能访问GitHub时，可在D盘新目录取得main（已有目录先检查Git状态，不能覆盖本轮离线代码目录）：

```bash
git clone --branch main https://github.com/Yiheng-Yuan/GRN-Synthetic-Perturbation.git /mnt/d/GRN/main-checkout
cd /mnt/d/GRN/main-checkout
```

本次实际因Windows访问GitHub超时，改用Mac打包的精确提交源码离线传输，运行目录为`/mnt/d/GRN/GRN-explicit-graph-990f8d8`。该目录不是Git工作区，无需git checkout。Windows上的SFTP文件传输落点是Windows文件系统；D盘绝对路径为`/D:/`，Ubuntu对应`/mnt/d/`，不能在该SFTP服务中把`/home/research/`当作目标目录。不要在已运行任务中替换源文件。

激活GRN环境并明确固定D盘输出，以下环境变量在新终端中需重新设置：

```bash
source /home/research/.venvs/grn-q1/bin/activate
export Q1_DATASET=/home/research/grn-q1-wsl-20261023/data/q1-modular-64
export Q1_OUTPUT_ROOT=/mnt/d/GRN/results
```

进入实际代码目录后，先运行最长时间的一次前向／反向GPU探测（当前任务已做过，不要重复启动）：

```bash
bash tools/scmultisim/run_q1_graph.sh probe
```

通过后启动完整四权重搜索与最终重拟合。以下是本轮已采用的后台启动方式，**仅执行一次**，已启动的任务不重复启动：

```bash
nohup bash tools/scmultisim/run_q1_graph.sh train \
  > /mnt/d/GRN/results/q1-graph-training.log 2>&1 < /dev/null &
```

如中断，配置、源码、设备与数据不变时：

```bash
bash tools/scmultisim/run_q1_graph.sh resume
```

默认数据：`/home/research/grn-q1-wsl-20261023/data/q1-modular-64`，清单SHA256绑定`7ce56f42b9b3ebe3206cb6591dba2536dfdb6511128cfce143a83751c97d50de`。可用Q1_DATASET调整目录但不更换数据。入口拒绝意外CPU训练；RTX4070已据用户日志通过新模型单次最长时间前向／反向检查，完整训练显存及收敛仍须实际运行。公共RNA数据另见[下载与内容说明](q1-public-data.md)。

本轮通过Q1_OUTPUT_ROOT指定的D盘完整输出（不是脚本未设变量时的默认目录）：

```text
/mnt/d/GRN/results/q1-graph-sweep-seed20261022/
  comparison_plan.json / selection.json / comparison.json / run.log
  sparse-0/ sparse-1e-5/ sparse-1e-4/ sparse-1e-3/
  final_refit/
```

Windows资源管理器对应`D:\GRN\results\q1-graph-sweep-seed20261022`。启动日志为`D:\GRN\results\q1-graph-training.log`。各训练目录完成后含model.pt、training_checkpoint.pt、learned_graph.npz、W/H/Z/probability.csv、efficiency.json、history.json、progress.jsonl、integration_check.json、run_plan.json、report.json、源码快照及SHA256SUMS.json；未完成时不能假定这些文件均已存在。CSV列为来源基因，数据第i行对应接收基因i（编号从0开始）；不包含模拟真图。final_refit另有scores.json及predictions.npz。probe独立保存在同一results下的`q1-graph-probe-seed20261022`。输出存在时拒绝覆盖；续跑核对源代码、数据、配置、环境和已完成文件。

可用`tail -n 30 /mnt/d/GRN/results/q1-graph-training.log`检查终端日志；每组progress.jsonl每次更新写入。初始图有4032个候选方向，不等于恢复了4032条真实边。

**回传只压缩完整的这一轮结果目录，不压缩代码项目、数据、虚拟环境或镜像：**

```bash
tar -czf /mnt/d/GRN/q1-graph-sweep-seed20261022.tar.gz \
  -C /mnt/d/GRN/results q1-graph-sweep-seed20261022
```

## 4. 独立图评分

评分侧显式提供真值文件，训练入口不接受真图参数：

```bash
PYTHONPATH=src python -m grn_experiment.q1_graph_evaluation \
  --graph /path/to/final_refit/learned_graph.npz \
  --truth /scoring-only/private/generator.json \
  --output /new/path/graph_scores.json
```

评分报告概率排序AP、二值边精确率／召回率／F1、假边／漏边、符号、边数及校验值。生产调控A与成熟RNA有效闭合图的差异必须保留；不比较不可比的H幅度或将成熟RNA导数冒充产生率Jacobian。测试通过、完成4800次更新或预测变好都不自动证明恢复了真实图。
