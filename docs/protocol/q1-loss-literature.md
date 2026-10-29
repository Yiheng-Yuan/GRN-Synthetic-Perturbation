# 实验一损失权重：四篇原始来源的核查与迁移边界

核查日期：2026-10-29。本文属于设计依据。随后已新增显式图开发实现与四权重运行器；用户回传GPU探测通过、第一组训练已启动。尚无完整候选比较结果，不冻结最优权重，也不以初期损失推断结构恢复。

## 1. 实际核对的证据

| 来源 | 与本项目有关的设置 | 求和／平均和适用边界 |
| --- | --- | --- |
| RegVelo 正式论文及作者代码 | 论文默认先验项系数1、Jacobian L1系数0；SDE实验另比较稀疏系数0、0.5、1。dyngen敏感性比较10^-4到10的十倍网格。 | 论文Jacobian项为逐细胞局部导数矩阵L1后取均值。固定版本代码在批次平均表达处计算Jacobian，其L1对矩阵元素求和；U/S重构负对数似然按基因求和、再按细胞平均。先验图惩罚、U/S重构和本项目的成熟RNA均值误差、二值边数不同。 |
| D-SPIN 正式论文方法索引及作者代码 | K562基因层网络L1为0.01、扰动场先验为3；程序层网络L1为0.01、场L2为0.005。作者代码默认网络L1为0.01、场L2为0.005；可选扰动先验默认0.5。 | 代码伪似然梯度在每条件内按细胞平均，再跨条件求和；网络L1梯度也逐条件计入。约束连续J权重；场先验是参数先验，不是RNA响应差误差。D-SPIN是平衡态统计模型，不直接提供本项目瞬态ODE损失。 |
| CLAMP 固定作者代码 | 配置稀疏系数5e-6、对照系数0.1；稀疏预热／渐增均为0。API默认稀疏系数1e-5、预热100／渐增50，与配置不同。 | 扰动项是冻结编码空间中KO−WT变化量的MSE，按条件和编码维度一起平均；稀疏项是有效邻接矩阵绝对值之和。并非二值W的边数，更不是本项目时序RNA群体分布损失。 |
| Shen等（2021）主文、补充材料和作者v1.0代码 | 主文轨迹目标为平方差求和后开方。适应性脚本将4个平方差求和项等系数相加后开方。主要通过逐边删除、重新训练精简网络，没有二值边数正则的通用系数。 | 普通神经网络参数不与调控边逐一对应，普通权重衰减未有效稀疏化其有效网络。补充材料另试局部作用幅度惩罚，不能将它误称为W边数。不同任务有不同监督项与求和／平均方式。 |

证据边界：RegVelo读取了正式论文相关方法页及固定代码；Shen读取了用户主文、官方补充和v1.0训练脚本，未执行作者训练。D-SPIN的论文方法数值来自PMC官方全文搜索索引；CLAMP原PDF下载遇访问验证，论文附录E.2第16页的`ctrl=1、sparse=1e-5`只由官方PDF搜索索引核对，未成功下载完整PDF复查。因此本文主要以直接读取的固定代码描述CLAMP的可核查配置，不能称这四篇均已完整精读。

CLAMP论文索引式（3）仅显示按条件平均平方欧氏范数；当前代码MSE还平均编码维度。即使惩罚对象相同，系数也不能忽略这个维度缩放直接比较。

## 2. 本项目采用的设计

保留原始要求`Z=W⊙H`，W决定有无边，H表示边参数。下一版拟采用：

\[
L=L_{\rm expression}+0.1L_{\rm distribution}
  +\lambda_W\sum_{i\ne j}P(W_{ij}=1).
\]

- 表达项：逐基因绝对表达的群体均值预测误差；处理与对照平衡计入，按已登记条件／时间／基因平均。表达缩放只用训练数据确定并固定，不随模型预测改变。
- 分布项：约束不配对细胞群体的异质性；具体距离、缩放与归约随模型实现登记。0.1是本项目的待验证候选，四篇文献没有验证我们这个项的0.1系数。
- 图项：随机门控时用期望边数之和，最终输出W必须二值。若使用不同可微松弛，另行登记与边数的对应关系。H或普通MLP参数的L1不能替代此项。
- 不恢复独立的处理减对照响应差损失。响应误差保留为独立评价指标。

第一轮以`lambda_W=1e-4`作为起始候选，固定分布权重0.1，比较`{0,1e-5,1e-4,1e-3}`共4组；之后补选中稀疏系数下的无分布项消融。这是借鉴文献的数量级比较与对照做法，**并非将任何作者系数直接移植为已验证答案**。不因真实生成网络有128条边而强制学习器输出128条。

尺度示例：64基因且无自环共有4032条可选方向。图项求和、系数1e-4时，100条确定存在的边贡献0.01，1000条贡献0.1。若改成边密度`mean(P(W=1))`，同等惩罚需将系数乘4032；因此单看系数大小或“是否全为1”没有可比性。

损失系数作为固定超参数，不由梯度学习。按训练靶点分组的内部验证选择候选；不使用已查看留出靶点、真实图或带稀疏罚项的训练总损失作权重选择标准。记录各未加权项、加权贡献、边数／门控概率，以及各参数组的梯度，检查过强稀疏或图通路失效。评价分别报告表达、分布、响应、边、符号与长期行为，不让训练权重改变最终评分定义。

实际开发实现将表达项定义为基于训练基线标准差（下限1）的两臂均值误差；分布项为基线固定log缩放下，逐基因与每轮重新抽取联合投影的平方Wasserstein距离各半。hard-concrete松弛用于拟合，最终预测／导出统一使用`P(active)>=0.5`的二值图。内部验证表达误差在绝对1e-6内视作平局，优先较少二值边、再较小稀疏系数。细节见[已实现入口](../q1-explicit-graph.md)。这些规则已写入软件；完整真实数据训练、GPU全流程验收、无分布消融与科学有效性尚未完成。

## 3. 可追溯来源

### RegVelo

- [正式论文PDF，方法e7与e18（完整PDF第36、47页）](https://push-zb.helmholtz-munich.de/deliver.php?id=45667)
- [固定代码ee8166e，参数默认值](https://github.com/theislab/regvelo/blob/ee8166ef3e341fc9d3a5ffb1104076725b130979/src/regvelo/_module.py#L435-L451)
- [固定代码ee8166e，loss实现](https://github.com/theislab/regvelo/blob/ee8166ef3e341fc9d3a5ffb1104076725b130979/src/regvelo/_module.py#L864-L912)

### D-SPIN

- [正式论文及STAR Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC13387596/)
- [固定代码f07f88e，默认正则参数](https://github.com/JialongJiang/DSPIN/blob/f07f88ef50a614c6a37212eba662f18cfeedf151/dspin/dspin.py#L239-L242)
- [每条件按细胞平均的梯度](https://github.com/JialongJiang/DSPIN/blob/f07f88ef50a614c6a37212eba662f18cfeedf151/dspin/compute.py#L664-L702)
- [正则梯度](https://github.com/JialongJiang/DSPIN/blob/f07f88ef50a614c6a37212eba662f18cfeedf151/dspin/compute.py#L1138-L1172)
- [跨条件求和](https://github.com/JialongJiang/DSPIN/blob/f07f88ef50a614c6a37212eba662f18cfeedf151/dspin/compute.py#L1293-L1304)

### CLAMP

- [论文PDF入口：本轮未能下载，附录数值仅由索引核对](https://openreview.net/pdf?id=C0DTxoMuYO)
- [固定代码389acde，实际配置](https://github.com/AmayaGS/CLAMP-Perturbseq/blob/389acde4913f72e8b0a970132cd7135f1b5dda68/configs/perturbseq.yaml#L58-L89)
- [MSE及API默认值](https://github.com/AmayaGS/CLAMP-Perturbseq/blob/389acde4913f72e8b0a970132cd7135f1b5dda68/perturbseq/ode_fitting.py#L573-L605)
- [稀疏项与总损失](https://github.com/AmayaGS/CLAMP-Perturbseq/blob/389acde4913f72e8b0a970132cd7135f1b5dda68/perturbseq/ode_fitting.py#L692-L712)

### Shen等（2021）

- [正式主文，特别是第3–4页](https://www.nature.com/articles/s41467-021-23420-5)
- [官方补充材料，S2、式S3及图S2](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-021-23420-5/MediaObjects/41467_2021_23420_MOESM1_ESM.pdf)
- [作者代码v1.0](https://github.com/sjx93/rnn_for_gene_network_2020/tree/v1.0)
- [固定代码归档Zenodo4705184](https://zenodo.org/records/4705184)：下载MD5与记录一致；代码只读未执行。
