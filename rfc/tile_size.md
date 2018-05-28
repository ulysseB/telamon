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
- dynamically add dimensions
  - at first, only expose the original dimensions, then add the tiles
  - must be accounted for by the performance model

To fid which one to use, we first need to develop them both.

## What is the tile size ?

TODO: do we expose tile sizes or just sizes
TODO: tile sizes or sizes identified by an id
TODO: mapped dims share the same size
