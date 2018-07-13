# Telamon

[![Build Status](https://travis-ci.org/ulysseB/telamon.svg?branch=master)](https://travis-ci.org/ulysseB/telamon)

Telamon is a framework to find good combinations of optimization for computational kernels
on GPUs. It currently focuses on dense linear algebra. For more information on how it
works internally, we encourage you to read our paper [paper][cc17].

## Getting Started

To compile Telamon, you need [rust 1.27 or higher](rust-install) installed. If you want to
generate code for GPU, you will also need a CUDA toolchain installed, with the `cuda`,
`curand` and `cupti` accessible in the include and library paths. You can the
[documentation on github][telamon-doc].

To generate code for GPUs, Telamon needs a description of the targeted GPU. This
description is generated with the following command:
```bash
$ cargo run --release --bin=cuda-characterize --features=cuda -- -w
```
that runs the `cuda-characterize` tool provided with Telamon. This command may take
several minutes to complete.

Examples of kernels are located in the `kernels/` directory. In particular,
`kernels/src/linalg.rs` contains linear algebra kernels. You can compare the code
generated by Telamon to the state of the art implementation on GPUs by running
```bash
cargo bench --features=cuda --bench=cuda-search
```
in the `kernel/` directory. To see the progress of the exploration, append
`RUST_LOG=telamon::explorer=warn` to the command.

## Writing a Kernel

To write a kernel, you must first define the inputs of the kernel and the context we
optimize for. Here, we assume we optimize for a Cuda GPU, but the process is the same
for other backends.

```rust
use telamon::device::cuda;
use telamon::helper;

let _ = env_logger::init(); // Enable logging
let executor = cuda::Executor::init(); // Setup the interface with the device.
// Build the signature and bind the inputs in the context.
let mut context = cuda::Context::new(&executor);
let array_a;
let signature = {
    let mut sig_builder = helper::SignatureBuilder::new("my_kernel", &mut context);
    // Create a signature with two arguments: a scalar `m` and an array of floats. We
    // give the value we want to optimize for to each argument.
    sig_builder.scalar("n", 1000i32);
    array = builder.array::<f32>("a", 1000); // Creates an array of size 1000.

    sig_builder.get()
};
```

We can now describe the body of the kernel itself. Here we create a kernel that computes
`x[i] = 2*i` for each `i in 0..n` For that we use a builder that creates the loops and the
instructions for us. The builder keeps the list of open loops and nest new instructions in
them.

```rust
let mut builder = helper::Builder::new(&signature, context.device());

// Open a loop of size n.
let size = builder.param_size("n");
let dim0 = builder.open_dim(size);
// Compute `x = 2*i` where `i` is the index on loop `dim0`.
let x = builder.mul(&dim0, &2i32);
// Store `x` in `a[i]. For that, we first compute the address of `a[i]` and build a
// that describes the access patern for the performance model.
let (addr, access_pattern) = builder.tensor_access(&"a", a, &ir::Type::I(32), &[&dim0]);
builder.st(&addr, x, access_pattern);

// Close the loop.
builder.close_dim(&dim0);

let search_space = builder.get();
```

We are now ready to start the search space exploration to find the best candidate.
```rust
use telamon::explorer;

let best = explorer::find_best(explorer::config::read(), &context, search_space).unwrap();
context.device().gen_code(&best, &mut std::io::stdout());
```

[rust-install]:(https://www.rust-lang.org/en-US/install.html)
[cc17]:(https://stratoss.fr/ulysse/papers/telamon_cc17.pdf)
[telamon-doc]:(https://ulysseb.github.com/telamon/telamon)