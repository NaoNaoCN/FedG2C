# Gefra Leave-One-Dataset-Out Protocol

`config/fedg2c_gefra_loo_v1.json` drives the Gefra cross-dataset leave-one-out
matrix through `python -m fedg2c.gefra_loo_matrix`. The runner keeps the four
graph-classification targets and all ten node-classification targets. For each
target it creates the explicit Gefra source-client list, runs FedG2C modules A
and B, then runs the four FedG2C target adaptations.

## Dataset Roles

Graph targets are ChemHIV, ChemPCBA, PROTEINS, and IMDB-BINARY. Each target
uses the other three datasets as one source client each. Node targets are
Cora, CiteSeer, PubMed, Photo, Computers, WikiCS, Arxiv, Reddit, Airport, and
Disease. The six core targets leave out one of the six core datasets. Gefra's
explicit three-source Arxiv and Reddit groups and six-source Airport and
Disease groups are retained as configured.

The source manifest omits labels. Graph targets use Gefra's deterministic
dataset-seeded random 80/10/10 partition. For ChemHIV and ChemPCBA, this is the
final split used by `graph_set_partition`, which replaces the official OGB
masks assigned by the raw loader. Node targets use Gefra's class-stratified ratios:
20/40/40 for Cora, CiteSeer, Photo, and Computers; 60/20/20 for PubMed; and
80/10/10 for WikiCS, Arxiv, Reddit, Airport, and Disease. All manifests record
the target identity and protocol seed. Source graphs without a non-self edge
cannot form FedG2C's edge-prediction view. The selected `exclude` policy omits
these source graphs and records every ID in the preparation report. Target
graphs remain complete.

Node source graphs retain their full features and edges, while roots without
a non-self incident edge are excluded from the traversal; their original node
IDs are recorded as well. All target nodes and target graph samples stay in
their original splits. Use the same protocol seed for the Gefra baseline:
matching split ratios alone does not guarantee matching train/valid/test IDs.

## Training

The matrix config uses 100 federated rounds, five complete local passes, and
one source dataset per client. Graph views accumulate independently in batches
of 64 before an optimizer and EMA update. Node epochs visit every non-isolated
source root once, in batches of 1,024 views. Module A remains FedG2C's
geometry-free experts, relation-conditioned composer, edge/feature corruption,
and nodewise EMA joint-embedding prediction. Module B remains FedG2C's SBM
output/finite-difference response/prior objective and its current ten server
optimization steps; Gefra's GW/Hungarian rule is not imported.

The shared source hidden dimension is 128 with two layers. Expert count follows
the number of source clients. Feature handling defaults to FedG2C-native; set
`matrix.feature_policy` to `gefra_compat` to use float OGB atom codes and
Gefra's 136-dimensional IMDB degree one-hot input. The feature policy is
included in each preparation identity.

Target epochs, learning rates, logical batch sizes, and validation intervals
come from Gefra's `finetune.yaml`. Graph targets select the minimum validation
BCE, preserving the agreed FedG2C rule; node targets select maximum validation
accuracy. The fixed support settings are full, 5-shot, and 1-shot. Each uses
the first sorted training IDs per class, and ChemPCBA takes the union across
task/class pairs. Ten target seeds are evaluated against the same source
checkpoint and split. Test evaluation is disabled in the tuning config; enable
it only in a separate final-selection run.

## Running

Preview one group or target before starting the full budget:

```bash
python -m fedg2c.gefra_loo_matrix --config config/fedg2c_gefra_loo_v1.json --group node --dry-run
python -m fedg2c.gefra_loo_matrix --config config/fedg2c_gefra_loo_v1.json --targets cora --dry-run
```

Run a chosen group with the full configured budget:

```bash
python -m fedg2c.gefra_loo_matrix --config config/fedg2c_gefra_loo_v1.json --group graph
python -m fedg2c.gefra_loo_matrix --config config/fedg2c_gefra_loo_v1.json --group node
```

Every target has a new data directory and run directory. Existing M1 runs,
shared dataset caches, and previous LODO outputs are not reused as writable
destinations. CPU is the matrix config's default device; change it to `cuda`
only on a host whose selected PyTorch environment reports CUDA available.

## Comparison Boundaries

This aligns dataset roles, source-client counts, split ratios, source passes,
main optimizer settings, target budgets, support selection, and validation
cadence. It does not turn FedG2C into Gefra: bounded edge-corrupted views,
geometry-free experts, response aggregation, and FedG2C's adaptation modules
remain in place. Node pretraining counts all eligible roots but samples each
root's bounded local view; it does not reproduce Gefra's PyG NeighborLoader
subgraph or full-graph target forward. Full target molecular graphs are retained
without cropping; their composer cost grows quadratically with graph size.
