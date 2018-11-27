import argparse
from pathlib import Path
import re

import yaml
import numpy as np
import pandas as pd
import statsmodels.stats.api as sms

def pop_kernel(config):
    axpy = config.pop('axpy', None)
    matmul = config.pop('matmul', [])
    no_matmul_fixed_tiling = config.pop('no_matmul_fixed_tiling', False)
    matmul_stride_a = config.pop('matmul_stride_a', None)
    batchmm = config.pop('batchmm', [])
    bmm_reuse_b = config.pop('bmm_reuse_b', False)
    bmm_static_sizes = config.pop('bmm_static_sizes', False)

    is_matmul = bool(matmul)
    is_batchmm = bool(batchmm)
    is_axpy = axpy is not None or (not is_matmul and not is_batchmm)

    if is_matmul + is_batchmm + is_axpy > 1:
        raise RuntimeError("Cannot be multiple kernels.")

    if is_matmul:
        if len(matmul) != 3:
            raise RuntimeError("Bad mamtul.")

        if matmul_stride_a == 32:
            stride = 'strided '
        elif matmul_stride_a:
            stride = 'strided({}) '.format(matmul_stride_a)
        else:
            stride = ''

        return 'generic {}matmul {}x{}x{}{} on f32'.format(
            stride, *matmul, ' (fixed tiling)' if not no_matmul_fixed_tiling else '')

    elif is_batchmm:
        if len(batchmm) != 4:
            raise RuntimeError("Bad batchmm")

        extra = []
        if bmm_static_sizes:
            extra.append('static sizes')
        if bmm_reuse_b:
            extra.append('reuse b')

        if extra:
            extra = ' ({})'.format(', '.join(extra))
        else:
            extra = ''

        return 'generic batchmm {}x{}x{}x{}{} on f32'.format(
            *batchmm, extra)

    else:
        assert is_axpy
        return 'generic axpy {} on f32'.format(axpy or 2 ** 26)

def pop_evaluator(config):
    evaluator = config.pop('evaluator', 'policy')
    stratifier = config.pop('stratifier', None)

    if evaluator == 'stratified':
        return 'Chen ({})'.format(stratifier)
    elif evaluator == 'policy':
        return 'Knuth'

    raise RuntimeError('Unknown evaluator {}'.format(evaluator))

def pop_cut(config):
    cut = config.pop('cut', None)
    max_cut_depth = config.pop('max_cut_depth', None)

    if cut is not None:
        return '{} (before {})'.format(cut, max_cut_depth)

    return None

def print_experiment(exp, *, level: int = 0, skip_custom_ordering=False):
    prefix = '  ' * level + ' * '
    if exp['ordering'] != ['lower_layout', 'size', 'dim_kind', 'dim_map', 'mem_space', 'order', 'inst_flag']:
        if skip_custom_ordering:
            return

        print(prefix + '**ORDERING**: {}'.format(exp['ordering']))

    #print(prefix + 'Found experiment {}/{} ({}; {} playouts with {})'.format(
    #    exp['directory'], exp['id'], exp['kernel'], exp['num_playouts'], exp['evaluator']))

    if exp['config']:
        print(prefix + "**** UNKNOWN PARAMS *****")
        print(yaml.dump(exp['config']))

    print(prefix + 'Size        : {:>7.2g}±{:>7.2g} ({}/{}; {} playouts with {})'.format(
        exp['mean'], exp['approx'],
        exp['directory'], exp['id'], exp['num_playouts'], exp['evaluator']))
    #if exp['cut']:
    #    print(prefix + 'Cut         : {}'.format(exp['cut']))
    #if exp['max_depth']:
    #    print(prefix + 'Before depth: {}'.format(exp['max_depth']))

def print_facets(facets, data, *, level: int = 0, **kwargs):
    prefix = '  ' * level + ' - '

    if isinstance(data, list):
        full = pd.concat([exp['df']['Estimate'] for exp in data])
        mean, approx = get_mean_approx(full)
        print(prefix + 'Total: {:>7.2g}±{:>7.2g}'.format(mean, approx))
        for exp in data:
            print_experiment(exp, level=level, **kwargs)

    else:
        for facet_value, facet_data in data.items():
            print(prefix + ', '.join(
                '{}: {}'.format(f, v)
                for f, v in zip(facets[level], facet_value)))

            print_facets(facets, facet_data, level=level + 1, **kwargs)

def get_mean_approx(series):
    lb, ub = sms.DescrStatsW(series).tconfint_mean()
    mean = (lb + ub) / 2
    approx = (ub - lb) / 2

    return (mean, approx)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('args', nargs='+')
    parser.add_argument('--skip-custom-ordering', action='store_true')
    parser.add_argument('--facet', dest='facets', nargs='*', action='append')
    parser.add_argument('--filter', nargs=2, action='append')
    ns = parser.parse_args()

    filters = {
        name: re.compile(value) for name, value in ns.filter or ()
    }

    experiments = []

    for arg in ns.args:
        for exp in Path(arg).glob('*'):
            if exp.joinpath('DUMMY').exists():
                continue
            if not exp.joinpath('estimates.csv').exists():
                continue

            config = yaml.load(exp.joinpath('config.yaml').open())
            if config.pop('dummy', True):
                raise Exception("Inconsistent dummy state.")
            if config.pop('prefix', []) != []:
                continue

            # Useless args for our purposes
            config.pop('cuda_gpus', None)
            config.pop('exact', None)
            config.pop('output', None)

            ordering = config.pop('ordering')
            num_playouts = config.pop('num_playouts')
            evaluator = pop_evaluator(config)
            kernel = pop_kernel(config)
            cut = pop_cut(config)
            max_depth = config.pop('max_depth', None)

            df = pd.read_csv(exp.joinpath('estimates.csv'), index_col=['Id'], dtype={'Estimate': np.float64})
            mean, approx = get_mean_approx(df['Estimate'])

            experiment = {
                'cut': cut,
                'id': exp.name,
                'config': config,
                'kernel': kernel,
                'num_playouts': num_playouts,
                'evaluator': evaluator,
                'directory': exp.parent.name,
                'ordering': ordering,
                'max_depth': max_depth,
                'mean': mean,
                'std': df['Estimate'].std(),
                'approx': approx,
                'df': df,
            }

            if all(value.search(str(experiment[name])) for name, value in filters.items()):
                experiments.append(experiment)

    kwargs = {
        'skip_custom_ordering': ns.skip_custom_ordering,
    }

    data = {}
    for exp in experiments:
        facet_data = data
        for ix, facet in enumerate(ns.facets):
            default = [] if ix == len(ns.facets) - 1 else {}
            facet_data = facet_data.setdefault(
                tuple(exp[f] for f in facet), default
            )
        facet_data.append(exp)

    print_facets(ns.facets, data, **kwargs)

if __name__ == '__main__':
    main()
