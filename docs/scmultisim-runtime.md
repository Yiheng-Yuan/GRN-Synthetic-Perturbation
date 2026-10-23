# 持续扰动扩展：运行与交付

**历史交付范围：** 下文 `20261014-dev1` 专指源码 `22b2fd7` 的 24 基因 CPU 生成镜像，不含后续修复或 Q1 训练器。当前 64 基因／H200 任务请使用[独立的新交付说明](q1-h200-delivery.md)，不得复用旧标签或覆盖旧归档。

本内核独立实现 scMultiSim 的双态启动子、未剪接 RNA（U）和已剪接 RNA（M）机制，以及本项目批准的固定背景 GRN 反馈、持续产生项敲低和不配对快照。语言是 Python，数值依赖是 NumPy；不调用官方 R 包，不把本扩展冒称官方 scMultiSim 1.8.0 的原生接口或逐值复现。主配置中的正式校准项仍保留 `null`。

## 直接运行

镜像名称 `grn-scmultisim`，版本 `20261014-dev1`，平台 `linux/amd64`。使用交付的镜像归档执行 `docker load -i grn-scmultisim-20261014-dev1-linux-amd64.tar.gz`，不需在计算节点下载 Python 包、代码或数据。

平台内启动命令：

```bash
python -m grn_experiment.generate_scmultisim --config /app/configs/scmultisim-development.json --output /output/scmultisim-dev-101
```

将可写的持久存储挂载到 `/output`。平台若使用不同挂载路径，只改 `--output` 后的路径；不要把结果留在任务结束后销毁的容器临时层。镜像默认命令与上面一致。

普通 Docker 启动：

```bash
mkdir -p outputs logs
docker run --rm --platform linux/amd64 \
  --mount type=bind,source="$(pwd)/outputs",target=/output \
  grn-scmultisim:20261014-dev1 > logs/generate.log 2>&1
```

### 每次交付都必须写明的回传范围

上面的示例将容器内 `/output/scmultisim-dev-101` 映射到宿主项目下 `outputs/scmultisim-dev-101/`，日志保存到 `logs/`。交给计算同学的说明须明确填写该平台上这两个目录的**实际绝对路径**，并附待回传文件名和打包命令。平台代管日志时，须导出任务的标准输出／标准错误到指定日志目录。

本任务在宿主项目根目录打包：

```bash
tar -czf scmultisim-dev-101-results.tar.gz \
  outputs/scmultisim-dev-101/ logs/
```

只回传这个结果包即可。`outputs/scmultisim-dev-101/` 内的 `manifest.json`、`observations/`、`sealed/`、`private/` 都属于完整生成结果；后两者由生成／评分侧保管，训练时另行隔离。Python 环境、缓存、镜像、源码均无需随每次结果重新压缩。一次任务若只完成部分条件，也保留相同回传范围及失败日志，便于依据清单恢复。

建议首次申请 2 个 CPU 核、4 GB 内存、5 GB 可写磁盘，任务时限先留 2 小时。这是预留配置，不是 H200 平台实测耗时。本程序单进程生成数据，不使用 CUDA 或 GPU；若平台必须分配一张 H200，该卡在本任务中不会参与计算。无需额外申请多卡。Slurm 分区、账号和平台镜像仓库命名由平台提供，不预设站点专属参数。

上传私有镜像库时，先在有 Docker 的设备加载归档，再按平台显示的完整 `docker tag` 和 `docker push` 命令操作。使用自己账号的仓库地址；本交付不包含任何账号密码或令牌。

## 本镜像生成什么

默认开发网络编号由 `network_seed=101` 确定：24 基因、16 调控者、40 条有符号边，`A[i,j]` 是 j 对 i 的作用。生成器共享一套网络、速率和每靶点效率；细胞随机性来自联合初态和启动子开关事件。

默认 `scope=pool`：

| 数据 | 数量 |
| --- | ---: |
| 共同扰动前基线 | 1 个键，3×128 个细胞 |
| 初始扰动条件 | 36 个 |
| 供策略选择的封存候选条件 | 90 个 |
| 封存验证条件 | 12 个 |
| 最终时序键 | 22 个扰动条件＋2 个晚时间对照 |
| 独立长期分布键 | 16 个靶点／剂量组合＋1 个对照 |

合计 163 个时序键、17 个长期键。每个靶向键为 3 重复，每重复 128 个处理细胞及 128 个独立对照。生成整个候选池不表示学习器可以免费读取它；每策略依然只能取得 12 个候选条件。

`scope=full` 可生成同一初始、验证和测试网格，但候选仅取均匀规则的 12 个，适合第一轮固定策略开发；`scope=initial` 只有共同基线和 36 个初始条件，且必须关闭长期评分。`configs/scmultisim-small-development.json` 将初始网格每重复细胞数缩为 4，用于本地软件演练；它不是交付镜像的默认任务。

所有扰动从 t=0+ 持续作用，t 仅表示无量纲采样时间。每次生成新条件、时间、重复或处理臂，都用独立随机流从同一未扰动联合分布重新生成完整 G/U/M 初态。最终只将 M 向下取整一次作为观测；没有额外技术噪声，不提供跨时间细胞配对。

## 具体实现与数值边界

`scmultisim.py` 提供完整参数、状态、分段解析解、精确指数等待时间、同步反馈和检查点。每个细胞／启动子有独立计数器随机流，事件序列不随采样请求或反馈分辨率改变。检查点保存参数、G/U/M、随机流键和计数器、待发生事件、保持的产生率、网格相位和当前请求时间。

反馈间隔内保持产生率，按事件解析推进 U/M；整个反馈网络仍是数值近似。观测时刻只读取副本，不改变后续路径。没有欧拉更新、负数截断或中间取整。极短分段及 β=γ 的解析极限有独立高精度参考测试。

开发映射明确选为 `g(z)=10^(1+0.25*z)`，`A` 的正负号决定局部产生率敏感度的符号。这是当前开发假设。生成入口从 101、102、103、104 四张开发参数参考的无反馈解析平均 M 中计算公共 K：所有参考基因平均 M 的中位数。映射、参考输入、计算规则和哈希均保存，未使用待测网络的反馈终点反求 K。显式 `parameter_file` 可载入另外登记的参数 JSON，此时必须自行提供已经确定的公共映射与 K。

正式映射配置采用 `{"kind":"positive_log_linear","log10_offset":1.0,"log10_slope":0.25,"scale":1.0}`，由实际 `PositiveMapping` 类直接保存／重建与验证；这个示例只展示格式，不代表当前三项数值已获正式校准。既有生成参数 JSON 的三个原始字段保持兼容。M-only 参数恢复还须遵守[等价类评分规则](protocol/parameter-equivalence.md)，不能将某个剪接／降解标签选取当作唯一真值恢复的必要条件。

`scmultisim_stationary.py` 对全关／零 RNA 与全开／高 RNA 两组独立谱系检查多个连续窗口，比较联合 G/U/M 的均值、方差、边缘及固定投影分布距离。默认最少运行 16、窗口长度 8、连续 3 次通过、最长 128；这些是开发判据，不是校准过的统计置信检验。生成细胞重新独立运行，不复用诊断池。初始分布不通过就停止；扰动长期条件不通过就登记失败，不输出冒充平稳的样本，不用 t=8 替代。

K、映射、最终反馈间隔、弱边门槛及分布容差还需在四张开发网络全面校准。当前只允许 `phase=development`；正式盲测生成被拒绝。单个四基因夹具的收敛检查不能替代正式校准。

## 输出、重启与状态

输出根目录是生成／评分侧的资料库，不能整个挂给学习器。

```text
manifest.json                 配置指纹、完成状态、条件记录与文件校验值
observations/initial/         共同基线与初始条件的 M 整数快照
sealed/candidates/           尚未付费取得的候选快照
sealed/validation/           第 12 次取得条件后才开放
sealed/final/                预测锁定后开放的最终时序
sealed/stationary/           单独锁定长期预测后开放
private/generator.json       参数、公共 K 参考、代码／运行时记录
private/truth/               初始和终点 G/U/M、局部产生率导数
private/provenance/          每个条件、臂、重复的私有随机种子
```

NPZ（压缩数组）观测文件只含 `treated` 和可选 `matched_control`，形状是 `(3, 128, 24)`。读取时用 `np.load(path, allow_pickle=False)`。私有真值的 `production_jacobian` 是在该快照连续 M 上计算的**未扰动** ds⁰/dM，与协议 J^s 一致；它不含该条件 KD 的乘数，也不等于仅观测 M 的马尔可夫方程导数。输入边权、局部导数与干预响应不能互相替代。

同一镜像、配置和输出路径重新运行时，会校验已经登记的文件并跳过完成条件。尚未完成的条件用相同私有随机流重做；正在生成的单个条件不会自动逐步落盘。内核 API 另外支持细粒度 JSON 检查点。源代码、参数、Python／NumPy 版本或配置变化会拒绝复用旧目录，防止混合不同版本的数据。

日志直接输出当前条件。退出码 0 表示完成；2 表示初始分布未通过；3 表示时序生成完成但有长期条件未证实平稳；1 表示运行错误。失败细节留在 `manifest.json`。对失败条件不自动更换网络或放宽判据。

生成器端可用 `load_blind_store(output_path)` 重建现有信息门禁；长期失败的资料库不会被当作完整盲测资料加载。程序门禁和目录名称不是跨账号安全沙盒，正式盲测还需独立评分进程与文件权限隔离。

## 从当前源码另建生成镜像

旧镜像复现必须使用 `22b2fd7` 的源码。下面从当前源码生成新标签，避免相同版本名对应不同实现；构建前应提交或另外归档工作树修改。

```bash
grn_image_revision=$(git rev-parse --short=12 HEAD)
docker build --platform linux/amd64 -f Dockerfile.scmultisim \
  -t "grn-scmultisim:dev-$grn_image_revision" .
docker save "grn-scmultisim:dev-$grn_image_revision" \
  | gzip > "grn-scmultisim-dev-$grn_image_revision-linux-amd64.tar.gz"
```

基础镜像固定到已拉取的摘要，Python 3.12.15，NumPy 2.2.6；运行时不联网安装依赖。`.dockerignore` 采用允许清单，排除数据、真值、结果、虚拟环境、Git 元数据及密钥文件。
