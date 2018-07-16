import os
from setuptools import setup

def env_true(name):
    return os.environ.get(name, '').lower() in ['true', '1', 't', 'y', 'yes']

def build_capi(spec):
    cmd = ['cargo', 'build', '--release']
    if env_true('TELAMON_CUDA_ENABLE'):
        print('CUDA build enabled.')
        cmd.extend(['--features', 'cuda'])

    build = spec.add_external_build(cmd=cmd, path='../telamon-capi')

    spec.add_cffi_module(
        module_path='telamon._capi',
        dylib=lambda: build.find_dylib('telamon_capi', in_path='../target/release'),
        header_filename=lambda: build.find_header('telamon.h', in_path='include'),
        rtld_flags=['NOW', 'NODELETE'],
    )

setup(
    name='telamon',
    version='0.0.1',
    packages=['telamon'],
    zip_safe=False,
    setup_requires=['milksnake'],
    install_requires=['milksnake'],
    milksnake_tasks=[
        build_capi,
    ],
)
