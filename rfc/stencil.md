# Summary

We propose to implement support for stencils in Telamon. Our approach should cover the
principal ways stencil can be implemented to achieve good performances.

# What is a stencil ?

Stencil are codes are codes that iterate on a neighborhood of each point of an array to
compute a new array. Both the array and the neighborhood can be multi-dimensional. In
practice this gives us a code that looks like the following:
```
for i in 0..N:
  for j in -K..K+1:
    out[i] += in[i+j]
```

The first question is how to handle the side conditions. Indeed, in the example above, we
access cells `in[-K]` and `in[N+K-1]` which are not defined. We usually have three
possibilities:
- Only produce the output if the input is defined. In practice, this means producing a
  smaller output.
- Use a default value when the input is not defined.
- Clamp the index between 0 and N-1, so that er use the border value multiple times.


# What do we want to generate ?

FIXME: how do we handle bounds ?

Stencil using rotating registers.
```
# Assuming K=1
x0 = in[-1]
x1 = in[0]
for i in 0..N:
  x2 = in[i+1]
  out[i] = x0 + x1 + x2
  x0 = x1
  x1 = x2
```

Stencil using a rotating memory. This version might not be needed. Notheless, we should
make sure it is possible to later add it.
```
array tmp[2K+1]
for j in 0..2K:
  tmp[j] = in[j]
for i in 0..N:
  tmp[(2K+i) % (2K+1)] = in[i+K]
  for j in -K..K:
    out[i] += tmp[(K+i+j)%(2K+1)]
```

Stencil implemented by unrolling everything.
```
# Assuming N=3 and K=1
x0 = in[-1]
x1 = in[0]
x2 = in[1]
x3 = in[2]
x4 = in[3]
out[0] = x0 + x1 + x2
out[1] = x1 + x2 + x3
out[2] = x2 + x3 + x4
```

FIXME: recompute border
FIXME: border exchange
FIXME: what is a border ?

```
for i in 0..N:
  for j in 0..M:
    x[i+M*j] = foo(i, j)
  for j in 0..M:
    for k in -K..K:
      out[i+M*j] += x[i+M*j+k]
```
