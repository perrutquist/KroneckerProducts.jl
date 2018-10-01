# KroneckerProducts.jl - Efficient Kronecker products in Julia

The [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) of
two matrices is a matrix whose size is the product of the sizes of the original
matrices.

Although it is a convenient mathematical concept, it is often inconvenient to
actually compute and store a Kronecker product in computer memory.

KroneckerProducts.jl provides the `⊗` operator, which creates `KroneckerProduct`
objects. Creating such an object requires almost no time or storage, as it
merely references the input arrays. The computation is performed each time an
element of the new array is accessed, and many operations can be performed
using specialized methods that do not require accessing all the elements of the
Kronecker product.

## Examples

The following code will create a `KroneckerProduct` matrix and then measure
the time it takes to multiply a vector by that matrix. (This takes advantage
of a specialized method for multiplying a vector by a Kronecker product.)

```
using KroneckerProducts
using BenchmarkTools
A = rand(20,20)
B = rand(30,30)
x = rand(600)
@btime ($a⊗$b) * $x
```

For comparison, we can use Julia's built in `kron` function to compute the Kronecker
product explicitly, and measure the time it takes to multiply a vector by that.

```
K = Base.kron(A,B)
@assert K == (a⊗b)
@assert K*x ≈ k*x
@btime $K*$x
```

(Exact timings will vary between systems, but the larger the matrices involved
the more advantageous it is to use `KroneckerProduct` instead of `Base.kron`,
and note that we did not even include the time it took to compute `K`.)

# Work in progress

There are not that many methods implemented at the moment. (Pull requests welcome!)
