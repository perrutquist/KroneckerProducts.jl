module KroneckerProducts

using Base.Cartesian
using LinearAlgebra

export Kronecker, KroneckerProduct

"""
A `Kronecker` object is a wrapper object that represents the applying a binary
operator to all combinations of the elements in two arrays in a fashion similar
to how the Kronecker Product works.

Creating a `Kronecker` object requires almost no time or storage, as it just
references the input arrays. The computation is performed each time an element
of the new array is accessed. Mutating the input arrays will mutate the result.
"""
struct Kronecker{T,N,F,A,B} <: AbstractArray{T,N}
    f::F
    a::A
    b::B
end

Kronecker{T,N}(f::Function,a::AbstractArray,b::AbstractArray) where {T,N} = Kronecker{T,N,typeof(f),typeof(a),typeof(b)}(f,a,b)

function Kronecker{T}(f::Function, a::AbstractArray{<:Any,NA}, b::AbstractArray{<:Any,NB}) where {T,NA,NB}
   N = max(NA,NB)
   Kronecker{T,N}(f,a,b)
end

function Kronecker(f::Function, a::AbstractArray{TA}, b::AbstractArray{TB}) where {TA,TB}
   T = typeof(f(zero(TA),zero(TB)))
   Kronecker{T}(f,a,b)
end

"""
A `KroneckerProduct` is a wrapper object that represents the Kronecker product
of two matrices (or the extension thereof for arrays in any dimension).

It is a special case of a `Kronecker` object.
"""
KroneckerProduct{T,N,A,B} = Kronecker{T,N,typeof(*),A,B}
KroneckerProduct{T,N}(a, b) where {T,N} = Kronecker{T,N}(*,a,b)
KroneckerProduct(a, b) = Kronecker(*,a,b)

Base.size(x::Kronecker{T,N}) where {T,N} = ntuple(i->size(x.a, i)*size(x.b, i), N)

@generated function Base.getindex(x::Kronecker{T,N,F}, idx::CartesianIndex{N})::T where {T,N,F}
    quote
        a = x.a
        b = x.b
        @nexprs $N i->(sz_i = size(b, i))
        x.f(@nref($N, a, i->(idx[i]-1)÷sz_i+1), @nref($N, b, i->(idx[i]-1)%sz_i+1))
    end
end

#Base.getindex(x::Kronecker, idx::Integer) = x[CartesianIndices(x)[idx]] # Not needed
Base.getindex(x::Kronecker, idxs::Integer...) = x[CartesianIndex(idxs)]

# Pass through wrappers from adjoint/transpose
Base.adjoint(x::KroneckerProduct{T,N}) where {T,N} = KroneckerProduct{T,N}(x.a', x.b')
Base.transpose(x::KroneckerProduct{T,N}) where {T,N} = KroneckerProduct{T,N}(transpose(x.a), transpose(x.b))

function LinearAlgebra.det(x::KroneckerProduct{T,2}) where {T}
    LinearAlgebra.checksquare(x)
    size(x.a,1) != size(x.a,2) && return zero(T)
    size(x.b,1) != size(x.b,2) && return zero(T)
    det(x.a)^size(x.b,1) * det(x.b)^size(x.a, 1)
end

function LinearAlgebra.tr(x::KroneckerProduct{T,2}) where {T}
    n = LinearAlgebra.checksquare(x)
    if size(x.a,1) == size(x.a,2)
        tr(x.a)*tr(x.b)
    else
        mapreduce(i->x[i,i], +, 1:n)
    end
end

gettmp(x::KroneckerProduct{Tx,2}, y::AbstractVector{Ty}) where {Tx, Ty} =
    Matrix{promote_type(Tx, Ty)}(undef, size(x.b,1), size(x.a,2))

function LinearAlgebra.mul!(y!::Vector, x::KroneckerProduct{<:Any,2}, y::AbstractVector;
        tmp=gettmp(x, y) )
    size(x,1) == length(y!) || throw(DimensionMismatch())
    size(x,2) == length(y) || throw(DimensionMismatch())
    mul!(tmp, x.b, reshape(y, size(x.b,2), size(x.a,2)))
    mul!(reshape(y!, size(x.b,1), size(x.a,1)), tmp, transpose(x.a))
    y!
end

end # module
