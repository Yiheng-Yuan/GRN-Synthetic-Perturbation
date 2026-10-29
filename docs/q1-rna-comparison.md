# 当前 RNA 与显式 RNA 历史：同数据开发比较

历史交付记录（2026-10-24）：下文是已经运行的无显式图对照，不是当前显式GRN主模型。当前入口见[W/H/Z开发模型](q1-explicit-graph.md)。镜像标签与校验值仅标识当时归档；从修改后的源码重新构建须使用新标签，不能覆盖旧镜像身份。

本次保留已经完成的含隐藏变量 h 的训练结果，另外训练两个模型。不重新生成数据，不使用真实 GRN、启动子状态、未剪接 RNA 或真实扰动效率。交付是 Linux/amd64（x86_64）CUDA 12.6 镜像；同一张 GPU 依次运行两组。另提供小型 WSL 原生代码包，供已配置好 PyTorch 的 Windows 笔记本直接使用。

## 模型与固定比较条件

| 项目 | 当前 RNA：`rna_current` | 显式历史：`rna_history` |
|---|---|---|
| 动态状态 | 64 个基因的非负 RNA 向量 x(t) | x(t) 及本次模拟已生成的 RNA 历史 |
| 变化规律 | dx/dt = f(x(t), u) | dx/dt = g(x(t), x(t−1), u) |
| 历史来源 | 无历史输入 | 模型自己的 RNA 预测；不读取真实扰动后历史 |
| 附加隐藏动态坐标 | 无 | 无 |
| 主网络宽度 | 64 | 51 |
| 参数数 | 12,609 | 12,720 |

网络内部的多层感知机激活只用于计算本次变化率，不会作为额外状态传到下一时刻。两组都直接保存 RNA，并使用恒等解码，避免把不能观测的负潜变量藏在截断解码器后面。扰动输入只在靶基因的局部分支直接作用，跨基因耦合由通用网络学习；不提供真实网络掩码。

历史延迟固定为 **1 个模拟时间单位**，不是“上一个采样时点”。最大数值步长为 0.25，默认整齐网格上对应四步前的 RNA；半步检查时仍保持延迟 1。t≤0 的历史设为该细胞抽到的基线 RNA 常量，这是一项初始化假设。每次模拟重新从 ctrl（对照）基线开始，历史不跨条件或训练更新保留。

两组使用同一非负投影 RK4（四阶龙格–库塔基础格式）推进：中间阶段和每步结束后都把负 RNA 投影到 0。接触边界时不能声称仍有四阶精度；历史插值也引入离散误差。模型必须通过最终步长减半检查；不通过则报告 `numerical_check_failed`，不宣称验收成功。

两组参数量相差约 0.88%，并非逐参数相同。当前组宽度 64、历史组宽度 51 是为缩小参数量差异预先固定的选择，不根据测试结果调整。保持原来的损失：24 个固定投影的群体分布距离、对照分布距离、标准化平均扰动响应误差；两组投影方向用独立于网络初始化的同一随机流生成，数据抽样顺序也相同。这轮不同时更改损失或扩大预算。

- 数据仍为 64 基因、48 个可扰动靶点；32 个训练靶点和 16 个留出靶点。
- 共同初态 t=0；训练时点为 0.25、1、2、4、8、64。t=64 是通过有限平稳性诊断后记录的长期快照，不是严格渐近稳态的证明。
- 每个条件三重复；只有前两重复拟合。训练靶点的第三重复用于检查，留出靶点用于评分。
- 每组 4,800 次参数更新；每次有效 4 条件、每条件 64 个抽样细胞，条件微批量 1，学习率 0.001，种子 20261022，每 60 次更新保存可恢复检查点。
- 每次从基线模拟到选定时间，比较整个细胞的 64 维 RNA 群体；没有按行配对真实相邻时间细胞，也没有真实一步变化率标签。
- 真实数据只有上述七个采样时点。数值积分在 8 到 64 之间推进，并不意味着这一区间存在额外观测。

现有留出结果已在旧 h 模型中查看过，因此这是同一开发网络上的方法比较，不是新的正式盲测。预测改善不能单独证明完整 GRN 恢复、真实单细胞轨迹恢复或生物系统的马尔可夫阶数。历史模型是否更好仍需要实际训练后的结果回答。旧 h 结果可作参考，但它使用不同状态表示、参数量及解码方式，不应把其差异全部归因于记忆。

## 文件

镜像标签：`grn-q1-rna:20261024-dev1`

- `grn-q1-rna-20261024-dev1-image.tar`：实际 Docker 镜像，包含学习器与启动脚本。
- `q1-modular-64-public.tar.gz`：与上轮相同的公共 RNA 数据，不含私有生成真值；已有相同数据可复用。
- `grn-q1-rna-20261024-code.tar.gz`：可选的 WSL 原生代码包；不含 PyTorch、数据或旧结果。
- `SHA256SUMS.txt`：以上交付文件的校验值。

镜像不包含数据、生成器、真实 GRN、旧模型权重或凭据。无需在运行时安装依赖或连接 GitHub。原镜像、旧数据和旧结果均不覆盖。

## Docker 启动：使用 Windows 上现有数据

下面在 WSL 的 Ubuntu 终端运行。先将新镜像放入 Windows 用户 Research 的下载目录。需要 Docker 的 GPU 接入已配置；此前成功的原生 PyTorch 检查不自动代表 Docker GPU 已接通。

```bash
docker load -i /mnt/c/Users/Research/Downloads/grn-q1-rna-20261024-dev1-image.tar
mkdir -p /home/research/grn-q1-rna-results
docker run --rm --gpus all --platform linux/amd64 \
  --mount type=bind,source=/home/research/grn-q1-wsl-20261023/data/q1-modular-64,target=/data/q1-modular-64,readonly \
  --mount type=bind,source=/home/research/grn-q1-rna-results,target=/output \
  grn-q1-rna:20261024-dev1 train
```

默认命令就是 `train`，会自动检查每一组在 t=64 的一个完整前向、反向、参数更新，然后丢弃检查权重，从登记种子重新开始正式的 4,800 次开发训练。检查成功后自动继续，不需人工再批准。它不使用测试靶点来决定能否启动，也不自动缩小数据量或改模型。

平台提交表如分别填写镜像与启动命令：镜像填上述标签（若上传到平台仓库，则用平台分配的镜像名），启动命令填 `bash /app/tools/scmultisim/run_q1_rna.sh train`。输入必须挂到 `/data/q1-modular-64`，可写输出必须挂到 `/output`。用卡需求为一张支持此 CUDA 运行时的 NVIDIA GPU；并行只跑一组。8GB 卡能否容纳本轮完整训练由本轮显存检查与实际运行确认，不能沿用旧 h 模型的显存结论。建议 4 CPU 核、16GB 主存，输出盘预留 2GB 以上。

中断后先确认旧训练进程已退出，使用同一条 Docker 命令，将末尾 `train` 换成 `resume`。它复核源码、数据、参数、环境和 GPU；已完整通过检查的一组跳过，未完成的一组从最近检查点恢复。没有检查点的残留输出不会被静默覆盖。若数值检查失败或配置不一致，保留日志发回分析，不以改名或删文件绕过验收。

## WSL 原生启动：已有 grn-q1 环境时更方便

无需为了本轮专门安装 Docker。把代码包放在 Windows 下载目录，然后执行：

```bash
source /home/research/.venvs/grn-q1/bin/activate
cd /home/research
tar -xzkf /mnt/c/Users/Research/Downloads/grn-q1-rna-20261024-code.tar.gz
cd /home/research/grn-q1-rna-20261024
sha256sum -c SHA256SUMS.txt
mkdir -p /home/research/grn-q1-rna-results
export Q1_DATASET=/home/research/grn-q1-wsl-20261023/data/q1-modular-64
export Q1_OUTPUT_ROOT=/home/research/grn-q1-rna-results
bash tools/scmultisim/run_q1_rna.sh train
```

中断后在相同环境与目录、保持上述两个环境变量，运行 `bash tools/scmultisim/run_q1_rna.sh resume`。Docker 与原生运行二选一；不要同时写入同一输出目录，不能跨环境续跑同一次训练。

若仅想先测显存，把 `train` 换成 `probe`；检查后用 `resume` 自动接着执行。直接 `train` 已含这个检查，不必重复手动运行。

## 唯一需要压缩回传的目录

**Windows／WSL 主机路径：**

`/home/research/grn-q1-rna-results/q1-rna-comparison-seed20261022/`

**Windows 资源管理器路径：**

`\\wsl.localhost\Ubuntu\home\research\grn-q1-rna-results\q1-rna-comparison-seed20261022`

容器内对应 `/output/q1-rna-comparison-seed20261022/`。如果同学使用其他主机输出挂载目录，则回传该挂载目录下同名文件夹。

内部包含 `run.log`、`comparison_plan.json`、两组显存检查、`rna_current/`、`rna_history/`，两组完成后还有 `comparison.json`。每组目录含模型、恢复检查点、训练损失、逐条件预测、评分、代码快照和校验值。最终评分和步长检查在 CPU 进行，GPU 利用率下降不代表进程已经结束。

只压缩这个结果目录：

```bash
tar -C /home/research/grn-q1-rna-results \
  -czf /home/research/q1-rna-comparison-results.tar.gz \
  q1-rna-comparison-seed20261022
sha256sum /home/research/q1-rna-comparison-results.tar.gz
cd /home/research
explorer.exe .
```

即使失败也回传同一目录。不必压缩项目、输入数据、Docker 镜像或虚拟环境。两组运行时间、峰值显存和预测效果以实际返回记录为准。
