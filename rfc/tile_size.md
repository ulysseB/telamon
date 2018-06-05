# Summary

We currently explore tile sizes by creating multiple search spaces. Instead we want to
expose tiling schemes as a regular decision. For this, we improve TelamonGen to expose
numeric decisons and to update the search space description.

# Improvements to TelamonGen

Tiles sizes can take a few values. However, the possible tile sizes are dependent on the
kernel instance and their is too many of them to statically enumerate them. Instead, we
need a set that can expose a few numerical values but where each of them is listed instead
of just taking the min and the max.

We just need to add a new kind of decisions.
```
define symmetric integer foo($lhs in A, $rhs in A) end
```

For that we need to:
- update the lexer, parser and AST
- update the constraints representation to manipulate integer decisions
- update the printer to print integer decisions

# Changes to the search space decision.

To handle tile sizes, we need to answer to explore two decisons:
- how many times is a dimension tilied ?
- what is the tiling factor ?

## How many times is a dimension tilied ?

We have identified multiple approaches:
- expose dimensions of size 1, but do not explore decisions on them.
  - increases the memory consumption
  - require logic to disable the decisions
  - must ensure we can avoid exploring the decisions and still generate code
  - cannot force the decisions to a particular value as it makes the logic too complicated
  - create a special set of such dimensions, that are not ordered
- dynamically add dimensions
  - at first, only expose the original dimensions, then add the tiles
  - must be accounted for by the performance model
    - have a flag that says if an instruction is nested in the all the subdimensions or
      only the outermost ?
  - all mapped dimensions must be lowered at once, and inserted in the new dim maps.

To find which one to use, we first need to develop them both.

TODO: choose an alternative to expose the "tiled or not tiled" decision

## What is the tile size ?

The sizes are used:
- To expose tile sizes
- To compute memory used (linked to tile sizes)
- For increments in memory accesses

We can either have a `size` choice for each dimension or have size IDs and reference them
from dimensions. In favor of the choice version:
- much simpler to implement
  - no size IDs
  - no need for
- expressive enough for tile sizes and memory used
In favor of the ID version
- mapped dims share the same size, they could share an ID
- might be easier to implement increment size

TODO: how do we link the size of the original dimension to the tile sizes ?

Another solution would be to only insert new dimensions once we know their size for sure.
This way, the size of dimensions is always the same. Problems:
- the size of the orginal dimension is still variable

TODO: choose a solution to expose tile sizes
