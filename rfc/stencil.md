# Summary

Stencils are a omnipresnet in computational kernels. We propose an approach to expose them
in Telamon.

# What do we want to generate.

Their is multiple way we can implement convolutions:
- with a loop to iterate on the neighborhood of a pixel
  - can be currently done with raw arrays
- by loading values in rotating registers
  - can be implemented with `fby`
- by exchanging the border of tiles
- by recomputing the border of tiles
