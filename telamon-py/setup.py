import json
import os
import subprocess
import sys

from setuptools import setup


def env_true(name):
    return os.environ.get(name, "").lower() in ["true", "1", "t", "y", "yes"]


def build_capi(spec):
    cmd = ["cargo", "+nightly", "build", "--release"]
    if env_true("TELAMON_CUDA_ENABLE"):
        print("CUDA build enabled.")
        cmd.extend(["--features", "cuda"])

    build = spec.add_external_build(cmd=cmd, path="../telamon-capi")

    out, err = subprocess.Popen(
        ["cargo", "+nightly", "metadata", "--format-version", "1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=None,
    ).communicate()
    if err:
        print(err, file=sys.stdout)
        return

    # Seriously, WTF milksnake...
    target_dir = json.loads(out.decode("utf-8"))["target_directory"]
    relative_target_dir = os.path.relpath(target_dir, build.path)

    spec.add_cffi_module(
        module_path="telamon._capi",
        dylib=lambda: build.find_dylib(
            "telamon_capi", in_path=os.path.join(relative_target_dir, "release")
        ),
        header_filename=lambda: build.find_header("telamon.h", in_path="include"),
        rtld_flags=["NOW", "NODELETE"],
    )


setup(
    name="telamon",
    version="0.0.1",
    packages=["telamon"],
    zip_safe=False,
    setup_requires=["milksnake"],
    install_requires=["milksnake", "toml", "numpy"],
    milksnake_tasks=[build_capi],
)
