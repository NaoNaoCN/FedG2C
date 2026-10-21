# FedG2C 阶段 A / B / C 代码时序图

打开 [fedg2c-stage-sequences.drawio](fedg2c-stage-sequences.drawio)，底部有 A、B、C 三个页面。每页上半部分为调度顺序，下半部分展开关键调用。

PNG 预览：[阶段 A](fedg2c-sequence-diagrams/preview-a.png) · [阶段 B](fedg2c-sequence-diagrams/preview-b.png) · [阶段 C](fedg2c-sequence-diagrams/preview-c.png)。可编辑源文件以 drawio 为准。

图中实线是调用，虚线是返回，折返箭头表示对象内部执行（或在本对象内调用标注的工具函数）；loop / alt / opt 表示循环、互斥分支和条件步骤。蓝色/紫色表示共享基座及参考模块，黄色表示本地可训练模块或优化器，绿色表示数据/结构函数。各页阶段标记按执行顺序编号。

源码位置采用 `相对路径:行号`，指向当前文件中的实际定义；同一函数中多个操作共享同一定义位置。鼠标悬停可查看完整类/方法名。对象生命线有时合并同类实例或工具函数组，已在参与者名称和旁注中说明；这种合并不表示实例共享权重。

## A：本地预训练

- A01–A07：建模、加载本轮共享权重、重置动量分支、创建本地优化器。
- A08–A28：采样双视图 → 可见图主分支 → 重算位置 → GraphSAGE/Transformer 预测 → 完整局部图动量参考 → Smooth L1。
- A29–A35：backward → clip → optimizer.step → EMA → 导出共享权重。
- AS01–AS14：专家逐一编码 → 结构关系 → 关系条件组合。

独立 A 入口：`pretrain_fedg2c.main --use-new-design` → `fedg2c.edge_entry.run`。联邦入口：`pretrain_federated_fedg2c.main` 在每轮、每个客户端执行 A，再执行 B。

## B：服务器聚合

- B01–B09：解包与校验；fedavg 提前返回；其他模式生成训练/诊断探针。
- B10–B23：缓存客户端参考 → 学习 logits → 可微混合候选参数 → 输出/响应匹配 → 更新聚合系数。
- B24–B27：最后一个优化步的聚合权重 → 独立诊断 → 保存 B 检查点。
- BE01–BE16：functional_call → 一次专家编码 → 基准组合输出 → 每个方向正负 gate 扰动 → 节点对余弦中心差分。

B 的基准权重和参考权重默认均匀，消息中的 num_samples=1 是兼容字段。output_only 的候选不测响应；参考缓存和诊断在关系分支可用时仍可测响应。扰动仅作用于 composer 的 relation_gate，专家编码和图边不变。

## C：目标域适配

- C01–C11：加载显式数据划分和 B 检查点 → 冻结原基座 → 安装三种适配之一 → 创建本地优化器。
- C12–C23：按 epoch/batch 训练；每个 batch 逐样本前向后拼接 logits；每个 epoch 验证并选择最低验证 loss。
- C24–C27：恢复最佳权重 → 校验冻结基座 → 一次最终测试 → 保存目标模型。
- CF01–CF19：私有输入适配 → 冻结专家 → 可选 Prompt → 关系组合器（可选 Q/K LoRA）→ 可选输出 Adapter → 可选本地编码融合 → 图/节点读出 → head。

| 方案 | 插入位置 | 额外训练参数 |
|---|---|---|
| Prompt | 专家输出 h[N,M,d] 加广播 prompt[1,1,d]，随后进入组合器 | prompt |
| LoRA | 组合器 query/key 的 Linear.forward 内 | Q/K 各自的 lora_A[r,d]、lora_B[d,r] |
| Adapter | 组合器输出 z[N,d] 之后 | down: d→b；up: b→d |

三种方案都保留目标域私有 node/edge adapter、分类 head，以及配置启用的 local_encoder / alpha_logit。默认 local_alpha=0。固定 local_alpha=1 为纯本地对照，跳过共享基座并冻结闲置的适配参数。

## 源码索引

- `main` — [aggregate_fedg2c.py:12](../aggregate_fedg2c.py#L12)
- `RelationComposer.forward` — [fedg2c/composer.py:47](../fedg2c/composer.py#L47)
- `head_rms` — [fedg2c/composer.py:14](../fedg2c/composer.py#L14)
- `hypernetwork_contract` — [fedg2c/composer.py:20](../fedg2c/composer.py#L20)
- `ContextSampler.__init__` — [fedg2c/context.py:67](../fedg2c/context.py#L67)
- `structural_relations` — [fedg2c/context.py:119](../fedg2c/context.py#L119)
- `EdgePredictionClient` — [fedg2c/edge_client.py:8](../fedg2c/edge_client.py#L8)
- `EdgePredictionClient.__init__` — [fedg2c/edge_client.py:10](../fedg2c/edge_client.py#L10)
- `EdgePredictionClient._summary` — [fedg2c/edge_client.py:75](../fedg2c/edge_client.py#L75)
- `EdgePredictionClient.get_local_message` — [fedg2c/edge_client.py:29](../fedg2c/edge_client.py#L29)
- `EdgePredictionClient.pretrain` — [fedg2c/edge_client.py:35](../fedg2c/edge_client.py#L35)
- `EdgePredictionClient.set_pretrain_model` — [fedg2c/edge_client.py:16](../fedg2c/edge_client.py#L16)
- `run` — [fedg2c/edge_entry.py:13](../fedg2c/edge_entry.py#L13)
- `MaskedEncodingBranch` — [fedg2c/edge_model.py:12](../fedg2c/edge_model.py#L12)
- `MaskedEncodingBranch.forward` — [fedg2c/edge_model.py:18](../fedg2c/edge_model.py#L18)
- `NodewisePredictionModel` — [fedg2c/edge_model.py:38](../fedg2c/edge_model.py#L38)
- `NodewisePredictionModel.__init__` — [fedg2c/edge_model.py:44](../fedg2c/edge_model.py#L44)
- `NodewisePredictionModel.forward` — [fedg2c/edge_model.py:81](../fedg2c/edge_model.py#L81)
- `NodewisePredictionModel.reset_momentum` — [fedg2c/edge_model.py:68](../fedg2c/edge_model.py#L68)
- `NodewisePredictionModel.update_momentum` — [fedg2c/edge_model.py:73](../fedg2c/edge_model.py#L73)
- `EdgePredictionPair` — [fedg2c/edge_views.py:15](../fedg2c/edge_views.py#L15)
- `NeighborhoodEdgeSampler` — [fedg2c/edge_views.py:58](../fedg2c/edge_views.py#L58)
- `NeighborhoodEdgeSampler.neighborhood` — [fedg2c/edge_views.py:65](../fedg2c/edge_views.py#L65)
- `NeighborhoodEdgeSampler.sample_edge_view` — [fedg2c/edge_views.py:80](../fedg2c/edge_views.py#L80)
- `mask_edges` — [fedg2c/edge_views.py:29](../fedg2c/edge_views.py#L29)
- `GraphSAGEPredictor` — [fedg2c/graph_predictor.py:12](../fedg2c/graph_predictor.py#L12)
- `GraphSAGEPredictor.forward` — [fedg2c/graph_predictor.py:22](../fedg2c/graph_predictor.py#L22)
- `client_message` — [fedg2c/messages.py:35](../fedg2c/messages.py#L35)
- `states_from_client_messages` — [fedg2c/messages.py:41](../fedg2c/messages.py#L41)
- `EdgePassingGraphEncoder.encode` — [fedg2c/model.py:44](../fedg2c/model.py#L44)
- `FeatureAdapter.forward` — [fedg2c/model.py:25](../fedg2c/model.py#L25)
- `SharedBackbone` — [fedg2c/model.py:56](../fedg2c/model.py#L56)
- `SharedBackbone.encode_experts` — [fedg2c/model.py:70](../fedg2c/model.py#L70)
- `SharedBackbone.forward` — [fedg2c/model.py:73](../fedg2c/model.py#L73)
- `NodeRepresentationPredictor.forward` — [fedg2c/node_predictor.py:31](../fedg2c/node_predictor.py#L31)
- `ResponsePreservingServer` — [fedg2c/response_server.py:34](../fedg2c/response_server.py#L34)
- `ResponsePreservingServer.diagnostics` — [fedg2c/response_server.py:100](../fedg2c/response_server.py#L100)
- `ResponsePreservingServer.evaluate_state` — [fedg2c/response_server.py:66](../fedg2c/response_server.py#L66)
- `ResponsePreservingServer.execute` — [fedg2c/response_server.py:107](../fedg2c/response_server.py#L107)
- `ResponsePreservingServer.matching_terms` — [fedg2c/response_server.py:78](../fedg2c/response_server.py#L78)
- `ResponsePreservingServer.objective` — [fedg2c/response_server.py:88](../fedg2c/response_server.py#L88)
- `ResponsePreservingServer.prepare_states` — [fedg2c/response_server.py:46](../fedg2c/response_server.py#L46)
- `ResponsePreservingServer.reference_cache` — [fedg2c/response_server.py:74](../fedg2c/response_server.py#L74)
- `distribution` — [fedg2c/response_server.py:14](../fedg2c/response_server.py#L14)
- `mix_parameters` — [fedg2c/response_server.py:21](../fedg2c/response_server.py#L21)
- `load_uploads` — [fedg2c/server_io.py:22](../fedg2c/server_io.py#L22)
- `save_aggregation` — [fedg2c/server_io.py:73](../fedg2c/server_io.py#L73)
- `ProbeEvaluator` — [fedg2c/server_probes.py:66](../fedg2c/server_probes.py#L66)
- `ProbeEvaluator.forward` — [fedg2c/server_probes.py:71](../fedg2c/server_probes.py#L71)
- `RelationProbe` — [fedg2c/server_probes.py:18](../fedg2c/server_probes.py#L18)
- `pair_readout` — [fedg2c/server_probes.py:60](../fedg2c/server_probes.py#L60)
- `sample_probe_bank` — [fedg2c/server_probes.py:32](../fedg2c/server_probes.py#L32)
- `random_walk_encoding` — [fedg2c/structural_encoding.py:12](../fedg2c/structural_encoding.py#L12)
- `TargetSample` — [fedg2c/target_data.py:11](../fedg2c/target_data.py#L11)
- `load_target_manifest` — [fedg2c/target_data.py:18](../fedg2c/target_data.py#L18)
- `ResidualAdapter.__init__` — [fedg2c/target_model.py:63](../fedg2c/target_model.py#L63)
- `ResidualAdapter.forward` — [fedg2c/target_model.py:70](../fedg2c/target_model.py#L70)
- `TargetModel` — [fedg2c/target_model.py:91](../fedg2c/target_model.py#L91)
- `TargetModel.__init__` — [fedg2c/target_model.py:92](../fedg2c/target_model.py#L92)
- `TargetModel.alpha` — [fedg2c/target_model.py:134](../fedg2c/target_model.py#L134)
- `TargetModel.forward` — [fedg2c/target_model.py:137](../fedg2c/target_model.py#L137)
- `TargetModel.frozen_base_unchanged` — [fedg2c/target_model.py:162](../fedg2c/target_model.py#L162)
- `TargetModel.parameter_report` — [fedg2c/target_model.py:165](../fedg2c/target_model.py#L165)
- `TargetModel.train` — [fedg2c/target_model.py:129](../fedg2c/target_model.py#L129)
- `install_lora` — [fedg2c/target_model.py:74](../fedg2c/target_model.py#L74)
- `load_shared` — [fedg2c/target_model.py:44](../fedg2c/target_model.py#L44)
- `classification_loss` — [fedg2c/target_train.py:14](../fedg2c/target_train.py#L14)
- `evaluate` — [fedg2c/target_train.py:67](../fedg2c/target_train.py#L67)
- `metrics` — [fedg2c/target_train.py:39](../fedg2c/target_train.py#L39)
- `predict` — [fedg2c/target_train.py:58](../fedg2c/target_train.py#L58)
- `run_target` — [fedg2c/target_train.py:73](../fedg2c/target_train.py#L73)
- `Linear.forward` — [fedg2c/vendor/loralib/layers.py:144](../fedg2c/vendor/loralib/layers.py#L144)
- `main` — [finetune_fedg2c.py:10](../finetune_fedg2c.py#L10)
- `GraphEncoder.forward` — [model/graph_encoders.py:153](../model/graph_encoders.py#L153)
- `main` — [pretrain_federated_fedg2c.py:24](../pretrain_federated_fedg2c.py#L24)
- `main` — [pretrain_fedg2c.py:10](../pretrain_fedg2c.py#L10)

生成器：[generate.py](fedg2c-sequence-diagrams/generate.py)。运行后会按 AST 重新解析当前源码的类/函数行号，并检查 XML、唯一 ID、边几何和页面数量。只读源码，不修改训练实现。
