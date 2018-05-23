# Introduction

This file purpose is to help newcommers navigate in Telamon's source code and to collect
the implicit rules about the code we write.

# Overview

The core Telamon library provides:
- a program representation, in `src/ir`, to describe the operations we want to implement,
- a search space representation, in `src/search_space`, to describe which optimisations
  are available,
- helpers to build program and search space instaces in `src/helper`,
- search space exploration algorithms in `src/explorer` and
- a performance model to help prune the search space in `src/model`.

The search space is described in a custom language that exposes decisions and the
constraint between them. This desription is then compiled with telamon-gen to generate
rust code to store and manipulate the search space. Integration tests for telamon-gen
itself are located in `telamon-gen/cc_tests`.

Generic function and algorithms go in the `telamon-utils` library. This library is meant
to be imported in each crate undef the `utils` name and then imported into each file with
`use utils::*;`.

Support tools are stored in the `tools/` folder. Currently, the two main tools are:
- `cuda-characterize`, that uses micro-benchmarks and public information to produce a
  description of the GPU architecture. This description is used to respect hardware
  constraints and model the performance of candidate implementations.
- `bench-perf-model` runs tests the accuracy and the correctness of the performance model.

# Coding Style

By default, we stick to the [official style guidelines][official_guide]. Here we only list
modifications and precisions:
* Lines should be at most 90 characters long.
* Every function, structure or trait definiton should have a corresponding comment.
* Every test or binary should start with `let _ = env_logger::try_init()` and include the
  [env_logger](crates.io/crates/env_logger) crate to enable logging.

[official_guide]:(https://github.com/rust-lang-nursery/fmt-rfcs/blob/master/guide/guide.md)
