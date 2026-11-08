"""Experiment roles/budgets copied from the inspected Gefra LODO configs.

These describe the comparison protocol, not Gefra's geometric model or its
GW aggregation. Expert/composer/predictor/response-server implementations stay
in FedG2C. Source order is significant: it fixes client IDs and random seeds.
"""

PROTOCOL_VERSION = 'gefra_loo_v1'
GRAPH_SOURCES = {
    'chemhiv': ['chempcba', 'imdb-binary', 'proteins'],
    'chempcba': ['chemhiv', 'imdb-binary', 'proteins'],
    'proteins': ['chemhiv', 'chempcba', 'imdb-binary'],
    'imdb-binary': ['chemhiv', 'chempcba', 'proteins'],
}
NODE_SOURCES = {
    'cora': ['citeseer', 'pubmed', 'photo', 'computers', 'wikics'],
    'citeseer': ['cora', 'pubmed', 'photo', 'computers', 'wikics'],
    'pubmed': ['citeseer', 'cora', 'photo', 'computers', 'wikics'],
    'photo': ['cora', 'pubmed', 'citeseer', 'computers', 'wikics'],
    'computers': ['cora', 'pubmed', 'citeseer', 'photo', 'wikics'],
    'wikics': ['cora', 'citeseer', 'pubmed', 'photo', 'computers'],
    'arxiv': ['photo', 'computers', 'wikics'],
    'reddit': ['cora', 'citeseer', 'pubmed'],
    'airport': ['cora', 'pubmed', 'citeseer', 'photo', 'computers', 'wikics'],
    'disease': ['cora', 'citeseer', 'pubmed', 'photo', 'computers', 'wikics'],
}

# epochs, Gefra early_stop (epochs; floor-divided by 10 in its logger),
# Gefra logical batch size (0 means all training roots), learning rate.
TARGET_BUDGETS = {
    'chemhiv': (400, 40, 1024, 5e-4),
    'chempcba': (30, 50, 512, 1e-4),
    'proteins': (200, 50, 128, 1e-4),
    'imdb-binary': (200, 50, 512, 1e-4),
    'cora': (1000, 50, 0, 1e-4),
    'citeseer': (1000, 300, 0, 1e-4),
    'pubmed': (1000, 200, 0, 1e-4),
    'photo': (1000, 200, 0, 1e-4),
    'computers': (1000, 200, 0, 1e-4),
    'wikics': (1000, 200, 0, 1e-4),
    'arxiv': (3000, 400, 0, 5e-4),
    'reddit': (10, 10, 0, 1e-3),
    'airport': (1000, 200, 0, 1e-4),
    'disease': (1000, 200, 0, 1e-4),
}


def sources_for(target):
    if target in GRAPH_SOURCES:
        return 'graph', list(GRAPH_SOURCES[target])
    if target in NODE_SOURCES:
        return 'node', list(NODE_SOURCES[target])
    raise ValueError(f'Unsupported Gefra graph/node target: {target!r}')
