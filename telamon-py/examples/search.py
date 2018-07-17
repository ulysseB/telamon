import argparse
import os
import telamon as tl
import toml

KERNELS = {
    'matmul': tl.MatMul(1024, 1024, 1024),
    'matmul-m-32-4-n-32-4-k-32': tl.MatMul(1024, 1024, 1024, m_tiles=[32, 4], n_tiles=[32, 4], k_tiles=[32]),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', choices=list(KERNELS), default='matmul')
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='gpu')
    parser.add_argument('--config', default=None)

    ns = parser.parse_args()

    config = None
    if ns.config is not None:
        with open(ns.config) as f:
            config = toml.load(f)

    kernel = KERNELS[ns.kernel]

    with tl.device(ns.device.upper()):
        kernel.optimize(config)
