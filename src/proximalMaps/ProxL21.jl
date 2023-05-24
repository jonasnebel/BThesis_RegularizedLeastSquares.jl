export L21Regularization, proxL21!, normL21

struct L21Regularization{T} <: AbstractRegularization{T}
  λ::T
  slices::Int64
  sparseTrafo::Trafo
end
L21Regularization(λ; slices::Int64 = 1, sparseTrafo::Trafo=nothing, kargs...) = L21Regularization(λ, slices, sparseTrafo)


"""
    proxL21!(x::Vector{T},λ::Float64; sparseTrafo::Trafo=nothing, slices::Int64=1, kargs...)

group-soft-thresholding for l1/l2-regularization.

# Arguments:
* `x::Array{T}`                 - Vector to apply proximal map to
* `λ::Float64`                  - regularization paramter
* `sparseTrafo::Trafo=nothing`  - sparsifying transform to apply
* `slices::Int64=1`             - number of elements per group
"""
proxL21!(x, λ; kargs...) = prox!(L21Regularization, x, λ; kargs...)
function prox!(::Type{<:L21Regularization}, x::Vector{T},λ::Float64; sparseTrafo::Trafo=nothing, slices::Int64=1, kargs...) where T
  if sparseTrafo != nothing
    z = sparseTrafo*x
  else
    z = x
  end
  if λ != 0
    proxL21!(z, λ, slices)
  end
  if sparseTrafo != nothing
    x[:] = adjoint(sparseTrafo)*z
  else
    x[:] = z
  end
  return x
end

function proxL21!(x::Vector{T}, λ::Float64, slices::Int64) where T
  sliceLength = div(length(x),slices)
  groupNorm = [norm(x[i:sliceLength:end]) for i=1:sliceLength]
  x[:] = [ x[i]*max( (groupNorm[mod1(i,sliceLength)]-λ)/groupNorm[mod1(i,sliceLength)],0 ) for i=1:length(x)]
end

"""
    normL21(x::Vector{T}, λ::Float64; sparseTrafo::Trafo=nothing, slices::Int64=1, kargs...) where T

return the value of the L21-regularization term.
Arguments are the same as in `proxL21!`
"""
normL21(x, λ; kargs...) = norm(L21Regularization, x, λ; kargs...)
function norm(::Type{<:L21Regularization}, x::Vector{T}, λ::Float64; sparseTrafo::Trafo=nothing, slices::Int64=1, kargs...) where T
  if sparseTrafo != nothing
    z = sparseTrafo*x
  else
    z = x
  end
  sliceLength = div(length(z),slices)
  groupNorm = [norm(z[i:sliceLength:end]) for i=1:sliceLength]
  return λ*norm(groupNorm,1)
end
