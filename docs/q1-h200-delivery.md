# Q1 64 基因 H200 训练交付说明

本任务使用已经生成的公共 RNA 数据，直接训练第一研究问题的潜变量神经常微分方程模型；不再生成数据，不执行主动选样，也不宣称恢复了真实 GRN。默认任务不是验证程序。

## 给计算端的内容

1. `grn-q1-modular-20261023-h200-dev1-image.tar`：训练镜像归档，导入后的标签为 `grn-q1-modular:20261023-h200-dev1`。
2. `q1-modular-64-public.tar.gz`：只有 `manifest.json` 与 `observations/q1/` 的公共数据包。解压顶层为 `q1-modular-64/`。不发送 `private/`、真实矩阵、参数或原项目目录。
3. 本说明及两份归档的实际 SHA-256 清单。具体归档大小和校验值必须在构建、打包完成后填写，不能用源码校验值代替镜像校验值。

资源申请：Linux `x86_64`、1 张 H200、4 个 CPU 核、16 GB 主存、10 GB 持久输出空间；镜像下载及解包另预留 30 GB。12 小时只是保守的任务申请上限，不是已经测得的 H200 训练时长。该小模型未必能充分利用整张 H200。

基镜像固定为官方 `pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime@sha256:a77983eb7a3042ccf41a94da547a7f54a47fb70da15c36d67a8e8a329ebaa098`。已只读核对官方清单为 `linux/amd64`，配置中包含 Python 3.12 路径及 CUDA 12.6.3；镜像构建再断言 Python ≥3.12、PyTorch 2.14.0，并固定 NumPy 2.5.3、pytest 9.1.1。运行时不下载依赖或访问网络。

基像信息核对不等于 GPU 验证。没有 H200 实测结果前，不得声称 CUDA 训练路径已经通过；真实 GPU 名称、运行时版本和设备检查结果由任务保存。宿主机必须提供兼容 CUDA 12.6 的 NVIDIA 驱动及容器 GPU 透传。

实际交付镜像现已构建，并通过容器内 138 项相关测试及真实数据的一次 CPU 前向／反向小检查；使用独立虚拟环境继承官方 torch，避免修改基础 Python 的系统包。归档实际大小、校验值和验证边界见[验收记录](validation/q1-modular-64-2026-10-23.md)。4800 次 H200 训练仍待本次任务执行。

## 容器平台填写

- 镜像：导入上述镜像归档后选择 `grn-q1-modular:20261023-h200-dev1`。
- 数据挂载：把解压后的公共数据目录挂到 `/input/q1-modular-64`，只读；其中 `/input/q1-modular-64/manifest.json` 必须存在。
- 输出挂载：把持久可写目录挂到 `/output`。
- 启动命令：`/bin/bash /app/tools/scmultisim/run_q1_h200.sh`。若平台尊重镜像入口，启动命令可留空。
- GPU 数量：1；不需要多卡、网络或分布式训练。

默认结果唯一目录为 `/output/q1-modular-64-h200-seed20261022/`。其中 `run.log` 是完整运行日志，`runtime.json` 是真实设备环境；模型及训练结果在 `results/`，包括 `model.pt`、`progress.jsonl`、`report.json`、预测、评分、校验清单及 `source_snapshot/`。若此目录已经存在，任务会拒绝覆盖；重试必须设置新的 `Q1_RUN_NAME=q1-modular-64-h200-第二次名称`。

脚本要求输出根目录预先存在且可写，不会在未挂载时自行创建默认 `/output`。目录存在本身不能证明它是持久存储，仍须在平台中确认真实持久化挂载，避免容器删除后丢失结果。

实际训练参数：48 个靶点的预登记 32／16 划分，训练只使用 32 靶点前两个重复，所有登记时点；4800 次更新、种子 20261022、CUDA、4 线程。模型隐藏宽度 64，RK4 步长 0.25；没有更换模型架构。最终模型固定后才评分留出靶点。长期观测仍采用数据中记录的真实有限采样时间，不能把它解释为每个细胞的零导数固定点。

## 普通 Docker 命令

在放有镜像和数据归档的目录核对实际校验清单后导入、解包；以下假定数据已解压，`persistent-output/` 已存在：

```bash
docker load -i grn-q1-modular-20261023-h200-dev1-image.tar
docker run --rm --pull=never --gpus '"device=0"' --network none \
  --read-only --tmpfs /tmp:rw,size=2g --shm-size=1g \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$(pwd)/q1-modular-64,dst=/input/q1-modular-64,readonly" \
  --mount "type=bind,src=$(pwd)/persistent-output,dst=/output" \
  grn-q1-modular:20261023-h200-dev1
```

`device=0` 只适用于已确认分配给本任务的宿主机设备；集群须使用平台分配的设备索引或 UUID，不能占用其他任务的卡。`tools/scmultisim/q1_h200.slurm` 是 Slurm 示例，分区、账号及 Docker 接入必须按实际平台配置；若平台自身已经启动容器，就使用上面的容器内启动命令，不在容器里再次启动 Docker。

## 只需回传什么

**只压缩这一个结果目录，不压缩项目、输入数据、镜像或环境：**

```bash
tar -C /实际持久输出目录 -czf q1-modular-64-h200-seed20261022-results.tar.gz \
  q1-modular-64-h200-seed20261022
sha256sum q1-modular-64-h200-seed20261022-results.tar.gz
```

请发回此结果归档及其 SHA-256。即使运行失败，也保留该目录中的 `run.log`、`runtime.json` 和失败记录；如果失败发生在容器启动前，另附平台任务的标准输出／错误日志即可。

## 构建端可选的极小 CPU 检查

这不是默认任务，也不代替正式训练。显式传入 `--cpu-smoke` 时，仅校验并读取公共基线和首个登记训练靶点 `t=0.25` 的两个 RNA 文件，执行一次 CPU 前向／反向更新，保存 `smoke.json`。不读取留出响应，不评分，不积分长期时点：

```bash
docker run --rm --pull=never --network none \
  --mount "type=bind,src=$(pwd)/q1-modular-64,dst=/input/q1-modular-64,readonly" \
  --mount "type=bind,src=$(pwd)/persistent-output,dst=/output" \
  grn-q1-modular:20261023-h200-dev1 --cpu-smoke
```

构建方法为 `docker build --platform linux/amd64 -f Dockerfile.q1 -t grn-q1-modular:20261023-h200-dev1 .`。专用白名单 `Dockerfile.q1.dockerignore` 排除数据、结果、Git 和私有真值；镜像只包含学习器、公开协议及测试，不包含生成器、真实网络或校准种子配置。
