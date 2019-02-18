#!/bin/bash

# Call this script to regenerate automated tests such as the NumPy format
# tests.

python src/tests/data/gen_npy.py src/ndarray_numpy_ext.rs
