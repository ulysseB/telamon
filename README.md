# Telamon

Telamon is a framework to find the best implementation of a basic dense linear
algebra function. It takes a description of the set of possible implementations
and evaluates them on a given input to find the optimal implementation.
Internally, it uses a performance model to trim the search space. The
performance model is guaranteed to never drop good candidates.

## Building

Telamon requires a nightly version of the rust toolchain. It can be installed with the
following command:

```bash
$ curl -s https://static.rust-lang.org/rustup.sh | sh -s -- --channel=nightly
```

Then git submodules should be initialized with the following command:
```bash
git submodule update --init --recursive
```

You can build Telamon using Rust package manager:

```bash
$ cargo build
```
the tests are run with:
```c
$ cargo test
```
and examples are built and run using:
```bash
$ cargo run --example=example_name --release
```

If you want to target NVidia GPUs, CUDA libraries (cuda, curand, cupti, cublas) must be
installed and available in the include and library paths (`C_INCLUDE_PATH`, `LIBRARY_PATH`
and `LD_LIBRARY_PATH`). To compile with cuda support, you must use the flag
`--features=cuda`.

To run, Telamon needs a description of the targeted GPU. The description can be generated
using the cuda-characterize tool. The compile and run the tool, use:
```bash
$ cargo run --release --bin=cuda-characterize --features=cuda -- -w
```
Characterizing the GPU may take a few minutes. You can follow the progress of
the characterization by prefexing the command with RUST_LOG=info.

## Structure of the Code

* `data/` contains the description of targets.
* `examples/` contains sample uses of Telamon to optimize basic functions.
* `src/` contains the Telamon framework.
  * `src/codegen/` handle code generation from a fully specified implementation
  * `src/device/` contains the target-specific code.
  * `src/explorer/` handles the search space exploration.
  * `src/helper/` provides builders to simplify the creation of an IR instance.
  * `src/ir/` contains the intermediate representation that allows representing sets of
  * `src/model/` contains the target-independent part of the performance model.
  * `src/search_space/` contains the search space description.
* `tests/` contains the integration tests.
* `telamon-utils/` defines generic helper functions.
* `telamon-gen/` contains the tool that generates the search space construction and
    representation code from a high-level description.
* `tools/cuda_characterize` is tool used to generate the description of a GPU. Only Kepler
  GPUs are supported for now.
* `tools/bench_perf_model` contains benchamrks to test the accuracy of the performance
  model.

## Coding style

If you want to make changes to the code, please respect the following rules:
* Lines should be at most 90 character long.
* Every function, structure or trait definiton should have a corresponding comment.
* Use `let _ = env_logger::try_init()` at the begining of tests and executables to enable
  logging.
