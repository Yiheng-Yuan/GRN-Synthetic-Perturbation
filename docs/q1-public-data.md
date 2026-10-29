# 64基因合成扰动开发数据

数据与代码分开保存。当前GitHub仓库公开，本附件只发布供学习器使用的合成RNA观测和开发元数据；不发布私有生成参数文件、真实GRN矩阵、G/U/M隐藏状态、训练权重或虚拟环境。开发数据曾用于模型比较，不能重新称为正式盲测。

## 下载与校验

- [发布页面](https://github.com/Yiheng-Yuan/GRN-Synthetic-Perturbation/releases/tag/q1-modular-64-data-v1)
- [RNA数据包](https://github.com/Yiheng-Yuan/GRN-Synthetic-Perturbation/releases/download/q1-modular-64-data-v1/q1-modular-64-public.tar.gz)
- [SHA256SUMS.txt](https://github.com/Yiheng-Yuan/GRN-Synthetic-Perturbation/releases/download/q1-modular-64-data-v1/SHA256SUMS.txt)

归档为11,292,437字节，SHA256：

```text
1175528cac46befbd1a97a2e4cf0e6dfd9d52731d1c0efc384ef9314457f3d09
```

解压后的manifest.json保留原始字节，SHA256为`7ce56f42b9b3ebe3206cb6591dba2536dfdb6511128cfce143a83751c97d50de`。当前训练运行器绑定这份清单，移动目录可以，修改清单不能冒充同一数据。

在可访问GitHub的终端下载（Windows无法访问时可先在Mac下载，再传到D盘）：

```bash
curl --fail --location --output q1-modular-64-public.tar.gz \
  https://github.com/Yiheng-Yuan/GRN-Synthetic-Perturbation/releases/download/q1-modular-64-data-v1/q1-modular-64-public.tar.gz
curl --fail --location --output SHA256SUMS.txt \
  https://github.com/Yiheng-Yuan/GRN-Synthetic-Perturbation/releases/download/q1-modular-64-data-v1/SHA256SUMS.txt
sha256sum -c SHA256SUMS.txt
```

## 包内内容

共290个普通文件：1份manifest.json及289份`observations/q1/*.npz`。

- 64基因、48个可扰动靶点，名义强度均为0.7。
- 共同t=0基线采一次；48个靶点各有t={0.25,1,2,4,8,64}，共288个扰动条件。
- 每条件3重复、每重复128个细胞。扰动条件包含`treated`和`matched_control`，形状均为(3,128,64)；共同基线只有`treated`。
- 共577个RNA数组，221,568个细胞观测行；不把这些行当独立网络样本。
- 数值为非负整数RNA计数；跨时间、处理和对照的行不配对，不能按同一行拼成单细胞轨迹。
- 32训练／16留出靶点登记在q1_split。新图模型在32内部再分24／8选超参数；原16已经查看，仅作开发比较。
- t=64是有限时间长期快照，在生成时通过相应分布诊断；不等同于严格渐近稳态证明。

清单保留开发诊断和私有文件的校验引用，不包含这些私有文件的载荷。历史重试错误字段也保留，以保持清单身份不变；当前发布的全部RNA载荷须以实际校验结果为准，而不是以历史错误字段判定缺失。逐文件审计见[公开数据发布检查](validation/q1-public-data-release-2026-10-29.json)。

## 解压与读取

在新目录解压，避免覆盖正在运行的原数据：

```bash
mkdir -p /mnt/d/GRN/data
tar -xzkf /mnt/d/q1-modular-64-public.tar.gz -C /mnt/d/GRN/data
```

结果是`/mnt/d/GRN/data/q1-modular-64/manifest.json`及其observations目录。当前已运行的任务继续用原数据路径，不在中途切换。

从仓库目录可以按清单读取（未安装项目时设置PYTHONPATH=src）：

```python
from pathlib import Path
from grn_experiment.q1_pilot_data import load_modular_q1

data = load_modular_q1(Path('/mnt/d/GRN/data/q1-modular-64'), scope='train')
print(data.baseline.shape)  # (3, 128, 64)
print(len(data.observations))  # 192 = 32靶点 × 6时点
```

`scope='train'`只打开原32训练靶点及共同基线，不读取16留出靶点的响应。完整开发读取可显式用`scope='all'`。逐文件内容和校验值以原清单为准；不要添加生成真值来绕过读取接口。
