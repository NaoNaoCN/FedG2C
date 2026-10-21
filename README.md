# Gefra

Gefra is a federated graph foundation model that combines multi-curvature
Riemannian mixture-of-experts encoders, geometry-aware federated aggregation,
and Riemannian Prompt Readout for downstream graph transfer.

Gefra provides a complete federated graph learning workflow:

- <code>pretrain_gefra.py</code> trains multi-curvature experts across source-graph clients.
- <code>finetune_gefra.py</code> adapts the aggregated model to node, link, and graph tasks.
- <code>load_gefra_model.py</code> restores a trained model for inspection.

Pre-training learns local graph representations on each client and aggregates
shared MoE experts and geometric projectors with Gromov-Wasserstein weighting
and Hungarian expert alignment. Fine-tuning fuses expert views through
Riemannian Prompt Readout and predicts labels with a linear task classifier.

## Repository map

| Path | Purpose |
| --- | --- |
| <code>pretrain_gefra.py</code> | Federated pre-training driver |
| <code>finetune_gefra.py</code> | Downstream fine-tuning driver |
| <code>load_gefra_model.py</code> | Model restoration and checkpoint inspection |
| <code>flcore/</code> | Federated clients, server aggregation, and optimization |
| <code>model/</code> | Graph encoders, Riemannian experts, projectors, prompt readouts, and task models |
| <code>task/</code> | Node, relation, and graph task training and evaluation |
| <code>utils/</code> | Dataset loading, deterministic splits, partitioning, logging, and utilities |
| <code>config/</code> | Per-target pre-training JSON and fine-tuning YAML settings |
| <code>scripts/</code> | Target-aware smoke, pre-training, fine-tuning, end-to-end, and robustness launchers |

## Environment

Create a Python environment with PyTorch, matching PyTorch Geometric wheels,
and the project dependencies. Select the training GPU with <code>--gpu_id</code>.

~~~bash
python -m pip install torch
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-$TORCH+$CUDA.html
python -m pip install -r requirements.txt
~~~

Set <code>$TORCH</code> to the installed PyTorch version and <code>$CUDA</code> to
the matching PyG wheel tag, for example <code>2.6.0</code> and <code>cu124</code>.

## Data and outputs

Datasets are cached under <code>&lt;project_path&gt;/cache_data/</code>. A symbolic
link can connect <code>cache_data</code> to shared dataset storage. Training writes
reusable artifacts to these locations:

- <code>partitions/</code> stores cached client partitions and split metadata.
- <code>logs/</code> stores training curves and metrics.
- <code>ckpts/pretrain_models/</code> stores server and client model weights.

Fine-tuning loads the server checkpoint created by the matching pre-training
configuration.

## Quick start

Run a complete Cora experiment:

~~~bash
bash scripts/end_to_end/run.sh cora
~~~

## Experiment launchers

Every launcher takes a target dataset as its first argument. The shared setup
discovers target configurations from <code>config/pretrain_config_*.json</code>,
derives the task family from <code>config/finetune.yaml</code>, and applies one
model configuration through both training stages.

| Script | Invocation | Workflow |
| --- | --- | --- |
| <code>scripts/pretrain/run.sh</code> | <code>bash scripts/pretrain/run.sh &lt;target&gt;</code> | Federated pre-training |
| <code>scripts/finetune/run.sh</code> | <code>bash scripts/finetune/run.sh &lt;target&gt; [k_shot ...]</code> | Fine-tuning from the matching checkpoint |
| <code>scripts/end_to_end/run.sh</code> | <code>bash scripts/end_to_end/run.sh &lt;target&gt; [k_shot ...]</code> | Pre-training followed by fine-tuning |
| <code>scripts/robustness/label_noise.sh</code> | <code>bash scripts/robustness/label_noise.sh &lt;target&gt; [k_shot]</code> | Label-noise robustness sweep |

Examples cover all task families:

~~~bash
# Node classification
bash scripts/end_to_end/run.sh cora

# Knowledge-graph relation classification
bash scripts/end_to_end/run.sh fb15k237 0

# Graph classification
bash scripts/end_to_end/run.sh proteins
~~~

Node and graph targets run full, 5-shot, and 1-shot fine-tuning by default:
<code>-1 5 1</code>. Link relation targets run the full edge split with
<code>-1</code> or zero-shot evaluation with <code>0</code>.

The launchers use the host <code>python</code> executable. <code>PYTHON_BIN</code>
selects a specific environment, and <code>RUN_ID</code> names each experiment
log directory.

~~~bash
PYTHON_BIN=python GPU_ID=0 STANDARD=1 RUN_ID=cora_trial_01 bash scripts/end_to_end/run.sh cora
~~~

The shared experiment variables are <code>GPU_ID</code>, <code>SEED</code>,
<code>ROUND</code>, <code>LOCAL_EPOCHS</code>, <code>STANDARD</code>,
<code>HIDDEN_DIM</code>, <code>TOPK</code>, <code>BACKBONE</code>,
<code>STRICT_PROJECTOR</code>, <code>ABLATION_FEDAVG</code>,
<code>ABLATION_NO_GEO_PROJECTOR</code>, <code>ABLATION_MOE</code>,
<code>PROJECT_PATH</code>, <code>PARTITION_ROOT</code>, <code>RUN_ROOT</code>, and
<code>DATA_CONFIG</code>. <code>K_SHOTS</code> sets a sequence of fine-tuning shot
values; <code>K_SHOT</code> and <code>NOISE_RATES</code> configure label-noise runs.

## Direct execution

<code>--tune_dataset &lt;target&gt;</code> selects
<code>config/pretrain_config_&lt;target&gt;.json</code>. Its <code>datasets</code> and
<code>pretrain_num_clients</code> fields define the source clients;
<code>finetune_datasets</code> and <code>finetune_num_clients</code> define the
target clients; and <code>num_experts</code> defines the multi-curvature expert
count.

### Pre-training

~~~bash
python pretrain_gefra.py --tune_dataset cora --gpu_id 0 --round 100 --local_epochs 5 --hidden_dim 128
~~~

The checkpoint path records source dataset order and client counts, backbone,
projector mode, ablation settings, and <code>topk</code>. Use the same
<code>hidden_dim</code>, <code>topk</code>, backbone, projector mode, ablation settings,
and <code>num_experts</code> for the subsequent fine-tuning run.

### Fine-tuning

~~~bash
python finetune_gefra.py --tune_dataset cora --gpu_id 0 --k_shot -1 --standard 1 --hidden_dim 128
~~~

Fine-tuning optimizes the private input projection, local encoder, prompt
readout, fusion weight, and linear task classifier. The aggregated MoE encoder
and geometric projector initialize the shared transfer representation.
<code>--ablation_learnable_weight</code> selects the learnable-weight readout. The
linear task head supplies both supervised logits and predictions.

<code>--k_shot -1</code> uses the full training mask, positive values select that
many examples per class, and <code>--k_shot 0</code> runs zero-shot evaluation.

Inspect a trained model with:

~~~bash
python load_gefra_model.py --tune_dataset cora --gpu_id 0
~~~

## Fine-tuning settings and data protocol

<code>config/finetune.yaml</code> stores four active settings for every dataset:
<code>finetune_epochs</code>, <code>early_stop</code>, <code>batch_size</code>, and
<code>finetune_lr</code>.

| Dataset family | Task | Effective split |
| --- | --- | --- |
| Cora, CiteSeer, Photo, Computers | Node classification | Deterministic class-stratified 20/40/40 |
| PubMed | Node classification | Deterministic class-stratified 60/20/20 |
| WikiCS, Reddit, arXiv, Disease, Airport | Node classification | Deterministic class-stratified 80/10/10 |
| WN18RR, FB15K237 | Relation classification | Class-stratified 80/10/10 edge split |
| ChemHIV, ChemPCBA, PROTEINS, IMDB-BINARY | Graph classification | Random 80/10/10 split for each client slice |

Relation classification represents an edge with the mean of its endpoint
embeddings and predicts its relation class. Graph classification pools node
embeddings into graph embeddings and reports AUC. Node and relation metrics
report accuracy.

### Label-noise robustness

Label-noise fine-tuning replaces a selected fraction of training labels while
keeping validation and test labels clean:

~~~bash
python finetune_gefra.py --tune_dataset cora --gpu_id 0 --k_shot 5 --standard 3 --ft_label_noise --ft_label_noise_rate 0.3
~~~

The label-noise launcher sweeps selected noise rates after one pre-training
run:

~~~bash
bash scripts/robustness/label_noise.sh cora 5
~~~

## Tests

Run the regression suite and entrypoint syntax checks with the same Python
environment used for experiments:

~~~bash
python -m unittest discover -s tests -v
python -m py_compile pretrain_gefra.py finetune_gefra.py load_gefra_model.py
~~~

Public Python comments and docstrings use clear English ASCII and explain
current behavior, invariants, and design choices.
