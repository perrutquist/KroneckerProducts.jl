using KroneckerProducts
using LinearAlgebra
using Test

a = rand(ComplexF64, 3, 2)
b = rand(ComplexF64, 4, 5)

k_big = kron(a,b)

k = KroneckerProduct(a,b)

@test k[1] == a[1]*b[1]
@test k[end] == a[end]*b[end]
@test k[1,1] == a[1,1]*b[1,1]
@test k[end,end] == a[end,end]*b[end,end]

@test k == k_big

@test k' == k_big'
@test transpose(k) == transpose(k_big)

@test size(k) == size(k_big)

@test eltype(k) == eltype(k_big)

y = rand(ComplexF64, size(k,2))
yr = rand(ComplexF64, size(k,1))

@test k*y ≈ k_big*y
@test yr'*k ≈ yr'*k_big

a = rand(3,2)
b = rand(2,3)
k_big = kron(a,b)
k = KroneckerProduct(a,b)

@test det(k) == 0
@test abs(det(k_big)) < eps()

@test tr(k) ≈ tr(k_big)

a = rand(3,3)
b = rand(2,2)
k_big = kron(a,b)
k = KroneckerProduct(a,b)

@test det(k) ≈ det(k_big)

@test tr(k) ≈ tr(k_big)
