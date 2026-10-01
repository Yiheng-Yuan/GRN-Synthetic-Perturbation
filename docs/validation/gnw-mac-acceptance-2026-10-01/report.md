# GNW 小网络适配验收

总体结果：通过

本报告仅验收一个确定性、RNA 层的小网络；不代表正式实验或生物学真实性验证。

| 检查 | 结果 | 数值证据 |
| --- | --- | --- |
| 原生 GNW 引擎与运行来源记录 | 通过 | {} |
| 未裁剪负值，未进行全数据归一化 | 通过 | {} |
| 两次原生运行的结果文件完全一致 | 通过 | {"file_count": 7} |
| 结果文件的实际校验值与记录一致 | 通过 | {"file_count": 7, "mismatches": []} |
| RNA 状态有限且非负 | 通过 | {"min_state": 0.0001341850497974278, "max_state": 0.5616752706935224} |
| 小网络轨迹数量及采样时间正确 | 通过 | {"paths": 53, "snapshots": 3445} |
| 所有条件共用扰动前初态 | 通过 | {"max_abs_error": 0.0} |
| 名义强度与实际抑制比例一致 | 通过 | {"max_abs_error": 0.0} |
| 同一靶点在不同强度、时间及初态下效率固定 | 通过 | {"max_abs_error": 0.0, "exception": "complete_ko deliberately sets G1 efficiency=1 for an analytic sanity check"} |
| 零剂量与对照轨迹重合 | 通过 | {"max_abs_error": 0.0} |
| 细化原生时钟及容差后轨迹一致 | 通过 | {"max_abs_error": 4.1524561567030105e-12, "threshold": 1e-07} |
| 独立重建函数的高精度积分与 GNW 一致 | 通过 | {"max_abs_error": 4.896860694714178e-12, "threshold": 1e-07, "integration_failures": []} |
| 原生变化速度及产生率与模块概率重建一致 | 通过 | {"rhs_max_abs_error": 0.0, "production_max_abs_error": 0.0, "probe_count": 68} |
| 原生产生率非负，零状态处不会向负丰度演化 | 通过 | {"min_production": 0.07332272727272723, "zero_boundary_min_derivative": 0.07599999999999998, "zero_boundary_probe_count": 17} |
| 扰动当下仅改变靶点产生项及其变化速度 | 通过 | {"direct_effect_max_abs_error": 3.0531133177191805e-16, "ratio_max_abs_error": 1.1102230246251565e-16} |
| 局部导数与原生导出及差分细化一致 | 通过 | {"max_abs_error": 0.0, "step_halving_error": 1.3877787807814457e-10, "threshold": 1e-08} |
| 边方向、促进抑制符号及非零实际作用正确 | 通过 | {"min_edge_local_effect": 0.04166666664473517, "nonedge_max_abs_error": 2.8755664516211255e-11} |
| 原生导出网络与预设矩阵一致 | 通过 | {"n_edges": 6} |
| 单独延长积分得到的稳态残差足够小 | 通过 | {"max_residual": 4.4131920340362285e-12, "stored_residual_max_abs_error": 0.0, "threshold": 1e-09} |
| 单独求得的稳态丰度有限且非负 | 通过 | {"min_state": 0.08681554590996991} |
| 多个对照初态经验性趋同 | 通过 | {"max_between_initials_error": 8.634204462509842e-12, "initial_count": 6} |
| 各扰动条件在多个初态下经验性趋同 | 通过 | {"max_between_initials_error": 8.961054120959489e-12, "condition_count": 16} |
| 后半程仍持续敲低，与根基因解析轨迹一致 | 通过 | {"max_abs_error": 4.1903702729939596e-12, "threshold": 1e-07} |
| 完全敲除后根基因按解析指数衰减 | 通过 | {"max_abs_error": 3.722772090597459e-12, "threshold": 1e-07} |
| 扰动确实传播到下游基因 | 通过 | {"max_downstream_response": 0.05693157099872059} |

t=8 最大变化速度残差：0.000292642；稳态另行求得，不将 t=8 当作稳态。

多初态趋同只是这些初态下的经验性证据，不是全局单稳态证明。

输出为连续 RNA 层有效丰度，不是测序整数计数；本轮不训练模型、不生成正式网络队列。
