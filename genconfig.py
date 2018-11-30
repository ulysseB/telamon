import copy
import toml
import pathlib

thresholds = [10, 50, 100]
deltas = [.1, 1., 10.]
factors = [1., 2., 6.]

base = {
    'max_evaluations': 100000,
    'algorithm': {
        'type': 'bandit',
        'tree_policy': {}
    },
}

tag_configs = []
for delta in deltas:
    for threshold in thresholds:
        tree_policy = {
            'type': 'tag',
            'threshold': threshold,
            'delta': delta,
        }
        config = copy.deepcopy(base)
        config['algorithm']['tree_policy'] = tree_policy

        tag_configs.append(config)

uct_configs = []
for factor in factors:
    tree_policy = {
        'type': 'uct',
        'factor': factor,
    }
    config = copy.deepcopy(base)
    config['algorithm']['tree_policy'] = tree_policy

    uct_configs.append(config)


def interleave(*its):
    while its:
        cur_its = its
        its = []
        for it in cur_its:
            try:
                yield next(it)
                its.append(it)
            except StopIteration:
                continue

tag_configs = iter(tag_configs)
uct_configs = iter(uct_configs)

base = pathlib.Path('experiments')
base.mkdir(parents=True, exist_ok=True)
for ix, config in enumerate(interleave(tag_configs, uct_configs)):
    expdir = base.joinpath(str(ix))
    expdir.mkdir()
    with expdir.joinpath('config.toml').open('w') as f:
        toml.dump(config, f)
