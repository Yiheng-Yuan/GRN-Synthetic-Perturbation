# GRN Synthetic Perturbation Experiment Framework

**Version 0.2.0** - Enhanced with Problem 2 implementation and configuration management

研究协议与可测试软件框架，用于在合成时序扰动系统中研究基因调控网络（GRN）的学习与辨识。

## 🎯 研究目标

### 问题一：未知动力学形式下的扰动传播学习
学习器不知道真实方程的解析形式，也不知道它属于哪些候选生成函数族。从有限的不配对 RNA 快照中学习有效的动力学 `F̂_φ(x, u)`，评估在未见条件下的 RNA 响应、GRN 边与符号、局部导数及长期行为。

### 问题二：已知方程架构下的参数可辨识性
学习器获得与生成系统一致的函数架构 `F(x, u; θ)`，但不知道真实参数、GRN 或靶点效率。研究对象是相容参数集 `Θ_k` 及其在主动实验下的收缩过程。

## ✨ v0.2.0 新增功能

### 核心实现
- ✅ **问题二完整实现**：`parameter_sets.py` 提供相容参数集构建、收缩指标和参数导向选样
- ✅ **配置管理系统**：`config.py` 实现类型安全的配置冻结和验证
- ✅ **效率学习机制**：`efficiency.py` 提供贝叶斯效率估计和响应校准
- ✅ **集成测试套件**：`tests/test_integration.py` 覆盖完整 12 轮选样循环
- ✅ **完整示例脚本**：`examples/run_complete_experiment.py` 端到端实验流程

### 改进
- 解决了硬编码问题，所有阈值现在通过配置管理
- 增强了信息隔离设计，明确标注评分侧/学习侧边界
- 添加了参数可辨识性分析工具
- 完善了文档和类型注解

## 📦 安装

```bash
# 克隆仓库
git clone <your-private-repo-url>
cd GRN-Synthetic-Perturbation

# 创建 Python 环境（推荐 3.12+）
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装开发依赖
pip install -e ".[dev]"
```

## 🚀 快速开始

### 1. 运行单元测试

```bash
# 运行所有测试
pytest -v

# 运行集成测试（较慢）
pytest tests/test_integration.py -v -s

# 只测试核心功能
pytest tests/test_simulation.py tests/test_protocol.py -v
```

### 2. 运行完整实验

```bash
# 创建输出目录
mkdir -p results

# 运行开发阶段（冻结配置）+ 2 个盲测网络
python examples/run_complete_experiment.py \
    --output results \
    --n-networks 2 \
    --problem both

# 只运行问题一
python examples/run_complete_experiment.py \
    --config results/frozen_config.json \
    --skip-development \
    --problem 1 \
    --n-networks 5
```

### 3. 使用 Python API

```python
from grn_experiment import (
    ExperimentConfig,
    make_network,
    make_dynamics,
    SnapshotSampler,
    SparseRNAODE,
    fit_unpaired_snapshots,
    ParameterSetApproximation,
)

# 1. 创建和验证配置
config = ExperimentConfig.from_defaults()
# ... 在开发网络上验证
config.save("frozen_config.json")

# 2. 生成网络和动力学
network = make_network(seed=42, n_genes=24, n_regulators=16, n_edges=40)
dynamics = make_dynamics(network, family="sigmoid", seed=123)

# 3. 采样观测数据
sampler = SnapshotSampler(dynamics, seed=456)
baseline = sampler.sample_condition(COMMON_BASELINE)
# ...

# 4. 问题一：训练学习器
model = SparseRNAODE(n_genes=24, min_efficiency=0.5)
fit_unpaired_snapshots(model, baseline, observations)

# 5. 问题二：构建参数集
param_set = ParameterSetApproximation(
    family="sigmoid",
    n_genes=24,
    bounds=ParameterBounds(),
    tolerances=tolerances,
    n_samples=1000,
)
param_set.refine(baseline, new_observations)
```

## 📊 模块架构

```
src/grn_experiment/
├── simulation.py          # 【评分侧】真实网络生成与动力学
├── observations.py        # 【评分侧】确定性采样器
├── protocol.py            # 【协议层】信息门禁与 BlindStore
├── learning.py            # 【学习侧】问题一：未知动力学学习
├── parameter_sets.py      # 【学习侧】问题二：参数集辨识 ⭐ 新增
├── baselines.py           # 【学习侧】对比基线模型
├── ambiguity.py           # 【学习侧】结构竞争者管理
├── selection.py           # 【学习侧】主动选样策略
├── efficiency.py          # 【学习侧】效率估计与追踪 ⭐ 新增
├── metrics.py             # 【通用】评分函数
├── workflow.py            # 【编排层】多策略实验管理
└── config.py              # 【配置层】参数冻结与验证 ⭐ 新增
```

### 信息流向（严格单向）

```
模拟器真值 → 采样器 → 协议门禁 → 学习器 → 评分
   ↑                                      ↓
   └─────────────── 只在评分时反向 ────────┘
```

## 🔬 实验设计

### 网络规模
- 24 个基因，16 个可扰动调控者
- 40 条有向边（入度≤3，出度≤5）
- 包含链、前馈、正负反馈基序

### 协议设计
- **6/6/2/2 调控者划分**：初始训练 6 / 主动候选 6 / 验证 2 / 测试 2
- **36 初始条件**：6 靶点 × 2 强度 × 3 时间
- **90 候选池**：6 靶点 × 3 强度 × 5 时间
- **12 次预算**：主动/随机/均匀三策略独立选样
- **3 重复**：前两个用于拟合，第三个用于检验

### 采样成本
- 共享基线（t=0）：384 细胞（128×3 重复）
- 每个靶向条件：768 细胞（扰动 384 + 对照 384）
- 12 次预算总计：9,216 细胞

## 📈 评估指标

### 问题一（未知动力学）
- **预测**：Sliced Wasserstein 距离、对照校正 RMSE
- **机制恢复**：边 AP、符号 F1、Jacobian RMSE、稳态误差
- **实验设计**：竞争结构排除比例

### 问题二（已知形式）
- **参数集收缩**：体积收缩率、存活分数
- **可辨识性**：参数组合区间宽度（如 amplitude × coupling）
- **预测分歧**：集合预测的分位数区间
- **真值恢复**：真实参数落入后验区间的覆盖率

## 🔐 信息隔离保障

### Python 私有字段（当前实现）
```python
class BlindStore:
    _observations: dict  # 命名约定，非访问控制
    _committed_predictions: dict | None
```

⚠️ **限制**：同进程代码可以访问 `_observations`

### 推荐部署（未来）
- **跨进程评分服务器**：学习进程通过 RPC 获取快照
- **文件系统权限隔离**：受限账户运行学习代码
- **容器隔离**：Docker/Podman 分离评分和学习环境

## 📝 配置管理

### 冻结配置（开发阶段）

```python
from grn_experiment import ExperimentConfig

# 1. 创建默认配置
config = ExperimentConfig.from_defaults()

# 2. 在开发网络上验证
diagnostics = {
    "all_capacity_passed": True,
    "all_weak_edges_acceptable": True,
    "all_steady_states_converged": True,
    "n_networks_checked": 8,
    "n_weak_edges": 12,
}
config.validate_on_development_networks(diagnostics)

# 3. 冻结配置
config.save("frozen_config.json")
```

### 加载冻结配置（盲测阶段）

```python
config = ExperimentConfig.load("frozen_config.json")
config.ensure_frozen()  # 确保已冻结，否则抛出异常

# 访问各子配置
print(config.network.n_genes)  # 24
print(config.protocol.active_budget)  # 12
print(config.learning.epochs)  # 250
```

## 🧪 测试覆盖

```bash
# 测试统计
tests/
├── test_simulation.py       ✅ 完整（网络生成、动力学）
├── test_observations.py     ✅ 完整（采样确定性）
├── test_protocol.py         ✅ 完整（门禁逻辑）
├── test_learning.py         ✅ 完整（前向传播、训练）
├── test_baselines.py        ✅ 完整（三种基线）
├── test_ambiguity.py        ✅ 完整（结构提议、资格检验）
├── test_selection.py        ✅ 完整（三种选样策略）
├── test_metrics.py          ✅ 完整（所有评分指标）
├── test_workflow.py         ✅ 完整（审计轨迹、多策略）
└── test_integration.py      ✅ 新增（端到端集成测试）

# 运行覆盖率报告
pytest --cov=grn_experiment --cov-report=html
```

## 🎓 使用场景

### 适合
- ✅ 方法探索和算法开发
- ✅ 单元级测试和基准比较
- ✅ 小规模原型验证（2-4 个网络）
- ✅ 教学和演示用途

### 不适合（当前阶段）
- ❌ 正式盲测实验（需要进程隔离）
- ❌ 大规模批量计算（需要分布式编排）
- ❌ 直接声称"已执行可信盲测"

## 🛣️ 开发路线图

### 已完成 ✅
- [x] 问题二的参数集构建与收缩
- [x] 配置管理系统
- [x] 效率学习机制
- [x] 完整集成测试
- [x] 端到端示例脚本

### 计划中 🚧
- [ ] 跨进程评分服务器（gRPC/ZMQ）
- [ ] 分布式任务编排（Ray/Dask）
- [ ] 效率估计的完整实现（与模型训练集成）
- [ ] 参数集的 MCMC 采样（替代网格近似）
- [ ] 更多合成数据源（scMultiSim 集成）
- [ ] 交互式结果可视化（Streamlit/Dash）

### 未来方向 🔮
- [ ] 真实单细胞数据适配
- [ ] 因果发现基准对比
- [ ] 多组学联合推断
- [ ] 在线学习和自适应设计

## 📚 相关论文

### 方法参考
- **主动实验设计**：Steiert et al. (2012) - 已知网络下的参数重拟合
- **竞争模型区分**：Mélykúti et al. (2010) - 信息性实验选择
- **单细胞模拟**：scMultiSim - 合成单细胞数据生成

### 生物学背景
- **扰动响应预测**：Perturb-seq, CRISPR screening
- **GRN 推断**：SCENIC, GRNBoost, CellOracle

## 🤝 贡献指南

本项目当前处于研究原型阶段。如有建议或发现问题：

1. 查看现有 Issues 是否已存在相关讨论
2. 创建新 Issue 描述问题或建议
3. Pull Request 欢迎（请先讨论重大变更）

### 代码风格
- 遵循 PEP 8 和类型注解
- 使用 Black 格式化（`black src/ tests/`）
- 运行 pytest 确保测试通过

## 📄 许可证

本项目用于研究目的。具体许可证待定。

## 🙏 致谢

感谢所有贡献想法和反馈的研究者。

---

**当前版本**：0.2.0  
**最后更新**：2024年10月  
**维护状态**：活跃开发中
