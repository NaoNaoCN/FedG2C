# 可审计的公开代码参考

这些是固定提交的官方原始文件，用于核查迁移内容，不是主程序运行依赖。原文件保持原样，因此不加 FedG2C 的新增区块注释。

| 来源 | 检查内容 | 项目接点 | 上游许可 |
|---|---|---|---|
| [BGRL](https://github.com/nerdslab/bgrl)，ICLR 2022 | `DropFeatures`、`DropEdges`、数据集增强率配置 | `fedg2c/edge_views.py`、`MaskedEncodingBranch` | [Apache 2.0](bgrl/LICENSE.md) |
| [GraphGPS](https://github.com/rampasek/GraphGPS)，NeurIPS 2022 | `get_rw_landing_probs` | `fedg2c/structural_encoding.py` | [MIT](GraphGPS/LICENSE) |
| [I-JEPA](https://github.com/facebookresearch/ijepa)，CVPR 2023 | `train.py` 的参考分支与动量更新；`VisionTransformerPredictor.forward` | `NodewisePredictionModel`、`NodeRepresentationPredictor` | [CC BY-NC 4.0](ijepa/LICENSE) |

下载地址、提交与每个文件的 SHA-256 见 [manifest.json](manifest.json)。代码测试核对快照哈希。

新增实现没有直接复制 I-JEPA 的源代码，参考其机制并使用 PyTorch 标准层独立实现；这里附带的上游快照仍按其原许可证提供。BGRL 的输入扰动被改写为局部张量接口；GraphGPS 的返回概率计算被适配为无权有界局部图版本。具体差异见 [本地预训练设计](../01_本地预训练_关系条件专家组合学习.md)。

既有超网络线性注意力的代码出处为 [官方固定版本](https://github.com/smonsays/hypernetwork-attention/blob/5dceaa1ebef5187a17a38d477a6b2fee8bc3dc38/hyla/models/attention.py)，对应 `fedg2c/composer.py`；许可声明另见 [third_party_notices](../third_party_notices)。本轮未修改该组合核心。

模块 B 的补充审阅快照与来源见 [module_b/README.md](module_b/README.md)。
