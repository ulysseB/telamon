# Summary

Telamon's IR currently represents values as instruction inputs with `ir::Operand`.
Unfortunately, this does not allows us to manipulate decisions about operands and to
reason about the data flow in the program. We propose to introduce a new type `ir::Value`
that represents a value stored in a variable.

# Motivation

Currently, the representation of values has the following drawbacks.
- We cannot link decisions to values, as they are hidden in operands. We at least want to
  expose decisions about:
  - where is a value stored
  - how many instances of the value are simultaneously stored
- We cannot easily detect where to introduce broadcasts. Instead we duplicate the
  computation on all threads.
- The user has to manually set decisions to ensure the generated code is correct. These
  decisions should be inferred at search space creation from the semantics of the IR.
- Reductions have a complicated representation and are not expressive enough. We need to:
  - express reductions in registers, with atomics and with messages; and
  - handle reductions that use multiple instruction.

# Detailed design

We create a new kind of object `ir::Value` with its associated ID `ir::ValueId` to
represent values created by instructions. Instead of directly referencing instructions IDs
with a dimension mapping, operands reference values IDs.

## Value Definition

The content of a value can be defined by:
- the instruction (`inst ir::InstId`) operator that returns the value produced by an
  instruction;
- the followed-by operator (`ir::ValueId fby ir::ValueId on Vec<ir::dim::Id>`), that
  returns the content of a first value at the first iteration of a loop nest and then the
  content of a second value, produced inside the loop nest;
- the last operator (`last ir::ValueId on Vec<ir::dim::Id>`), that returns the last
  content taken by a value in a loop nest; and
- the dimension mapping operator (`map ir::ValueId on Vec<(ir::dim::Id, ir::dim::Id)>`),
  that returns a pointwise mapping of the content of a value produced in another loop
  nest.

Each value has a type, which is either the type produced by the instruction or by the
values it depends on. Because instructions are not used to represent values anymore, we
can remove the `Void` type.

## Production dimensions

The iteration dimensions of a value are the dimensions of the loop nest in which the
value is defined.
- The iteration dimensions of `inst i0` value are the iteration dimensions of `i0`.
- The iterations dimensions of `v0 fby v1 on [d0, .., dn]` are the iteration dimensions of
  `v1`. We require that `iter_dims(v0) + [d0, .., dn] = iter_dims(v1)`.
- The iterations dimensions of `last v0 on [d0, .., dn]` are the iteration dimensions of
  `v0` minus `d0, .., dn`.
- The iteration dimensions of `map v0 on dims [(d0, d0'), .., (dn, dn')]` are the iteration
  dimensions of `v0`, where `di` is replaced by `di'`, forall i in 0..n.

A value can only be used by an instruction if the instruction is nested in all the
iteration dimensions of the value.

TODO: what if we want to broadcast before applying a forall ?
- we could define the iteration dims of `v0 fby v1` as the iteration dims of `v1` and
  rely on lowering to insert the broadcasts as necessary ?
TODO: fby production dimensions do not reflect on which dimensions do we need to broadcast
the init value. Indeed it may be nested in more dimensions during exploration.
- We could have the notion of values nested in a dimension (but this is quadratic, and
  will require a quotient in the search space).
TODO: are iteration dimensions fixed during exploration ?
- con: fby values behave as if iteration dimensions are added when nested outside
- con: the production dimensions of instructions vary, so why not the dimensions of values ?.
- pro: if we change the production dimensions, we might need to add last operators on the
  fly.

## Data Dependencies

Alongside iteration dimensions constraints, instruction consuming values must respect data
dependencies. Four kinds of events can happen on values:
- a value can be produced,
- a value can be read,
- a value can be destroyed and
- a value can be consumed.
The storage of a destroyed value may be overwritten by an other value. This means values
cannot be read after they are destroyed. A consumed value may also be overwritten, but
must otherwise keep holding the value. Thus, consumptions acts simultaneously as a read
and a destruction, with the additional constraint that a value may only be consumed once.

- inst `i0` is produced at `i0`
- `last v0 on [dims]` reads `v0` and shares the production destruction and consumption
  events with `v0`.
- `map v0 on [dims]` shares the production destruction and consumption events with `v0`.
- `v0 fby v1 on [dims]` consumes `v0` and is destroyed when `v1` is produced

When consuming a value, one should be careful that the value cannot be broadcasted across
a dimension. Indeed, this would result in the value being consumed multiple times.

TODO: find a better formulation

## Value Locations

We must map values to the actual location. While we might not need to know the exact
register name, we at least need to know in which memory space the value will be stored
(register, local, shared, global, ...) to generates the required memory acceses. We also
need to account for the amount of memory used.

Rather than tracking where the communication between operators happends, we track where
the operators actually happens.

We need to track:
- which values are mapped to the same location.
- which values account for a memory use and which already are accounted for
  > can be done statically (if clones are static) with a flag
- where are the values mapped

TODO: who is this encoded in the search space

## Broadcasting

There is two situations in which we must broadcast. The first one is when a value is
consumed along a loop which is not a production dimension, in which case, a copy must be
introduced. The second one is when a value must be sent to other threads because it is
stored in a local memory not accessible to others.

TODO: How do we enforce this ?
TODO: How do we ensure we only broadcast once ?
TODO: Can we always implement local and thread bcasting in the same way ? Or should they
 be implemented in two different ways ?
TODO: What is interesting for broadcasting is value use, not values themselves. Should
  broadcasting be handled at the value level ?

## Reductions

TODO: what do we want ?
- in place reduction in a register, iterated in a loop
- unrolled reduction, with reduction tree at the end
  > can we express this with `fby` ?
  > is it usefull ?
- reduction tree accross threads, with communications
- atomic update in memory

TODO: How can we implement reductions ?
how to express global sync ?
  > need to express patterns such as below
    dim 0.0
        def x
    dim 1.0
      dim 0.1
        dim 0.2
          y = red x along 0.2
        x = y
    dim 0.3
      proj x along 1.0

# Implementation

TODO

A value needs to store:
- Its type
- Its iteration dimensions (?)

## Impact or the Performance Model

TODO

## Impact on Code Generation

TODO

## Implementation Steps

TODO

We plan to perform the implementation of the new value system with the following steps:
- Create a value type that only represent the `inst` operator and use it in operands
- Remove the void type
- Add a last operator and enforce iteration dimension rules

# Open questions and alternatives

- How do we keep IDs coherent when lowering values ?
- Are internal arrays values ?
  - They are issued from values and behave as such
- Are external arrays values ? If internal array are values, why not external arrays ?
- are constants and indexes values ?
  - this depends whether `mov/copy` is considered as an instruction. If yes, then constants
  are not values since they must be passed into a mov to be implemented as a variable.
  - we could have a `nowhere` location for constant, that must be copied if used in
    fby/proj

TODO
