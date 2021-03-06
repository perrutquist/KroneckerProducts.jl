# KroneckerProducts.jl

The [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) of
two matrices is a matrix whose size is the product of the sizes of the original
matrices.

Although it is a convenient mathematical concept, it is often inconvenient to
actually compute and store a Kronecker product in computer memory.

KroneckerProducts.jl defines the `⊗` operator to create lazy `KroneckerProduct`
objects. Creating such an object requires almost no time or storage, as it
merely references the input matrices. The computation is performed each time an
element of the new matrix is accessed, and many operations can be performed
using specialised methods that do not require accessing all the elements of the
Kronecker product.

At the Julia REPL and in editing envrionments like Juno, the character "`⊗`"
can be typed as "`\otimes`" followed by <tab>. The unexported
`KroneckerProducts.kron` function is an ACII-compatible alternative.

## Example

The following code will create a 600-by-600 `KroneckerProduct` and then measure
the time it takes to multiply a vector by that. (This takes advantage of a
specialised method for multiplying a vector by a Kronecker product.)

```
using KroneckerProducts
using BenchmarkTools
A = rand(20, 20)
B = rand(30, 30)
x = rand(size(A,2)*size(B,2))
@btime ($A ⊗ $B) * $x  # approximately 5 μs
```

For comparison, we can use Julia's built in `kron` function to compute the
Kronecker product explicitly, and measure the time it takes to multiply a vector
by that.

```
K = Base.kron(A, B)
@assert K == A ⊗ B
@assert K * x ≈ (A ⊗ B) * x
@btime $K * $x  # approximately 16 μs
```

Exact timings will vary between systems, but the larger the matrices involved,
the more advantageous it is to use a `KroneckerProduct` instead of `Base.kron`,
and note that we did not even include the time it took to compute `K`.

# Work in progress

There are not many specialised methods implemented at the moment.
(Pull requests are welcome!)
