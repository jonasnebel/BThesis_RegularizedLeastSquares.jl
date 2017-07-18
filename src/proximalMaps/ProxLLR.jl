export proxLLR!, normLLR

@doc """
proximal map for LLR regularization using singular-value-thresholding" ->

### parameters:

* λ::Float64: regularization parameter (threshold)
* shape::Tuple{Int}: dimensions of the image
* blockSize::Tuple{Int}: size of patches to perform singluar value thresholding on
""" ->
function proxLLR!(reg, x)
  x[:] = proxLLR(x, reg.params[:lambdLLR]; reg.params...)
end

function proxLLR{T}(x::Vector{T}, λ::Float64=1e-6; shape::NTuple=[], L=1, blockSize::Array{Int64,1}=[2; 2], randshift::Bool=true, kargs...)
  xᵖʳᵒˣ = zeros(T,size(x))
  N = prod(shape)
  K = floor(Int,length(x)/(N*L))
  for i = 1:L
    xᵖʳᵒˣ[(i-1)*N*K+1:i*N*K] = vec( svt(x[(i-1)*N*K+1:i*N*K], shape, λ; blockSize=blockSize, randshift=randshift, kargs...) )
  end

  return xᵖʳᵒˣ
end

function svt{T}(x::Vector{T}, shape::Tuple, λ::Float64=1e-6; blockSize::Array{Int64,1}=[2; 2], randshift::Bool=true, kargs...)

  x = reshape( x, tuple( shape...,floor(Int64, length(x)/prod(shape)) ) )

  Wy = blockSize[1]
  Wz = blockSize[2]

  if randshift
    srand(1234)
    shift_idx = [rand(1:Wy) rand(1:Wz) 0]
    x = circshift(x, shift_idx)
  end

  ny, nz, K = size(x)

  # reshape into patches
  L = floor(Int,ny*nz/Wy/Wz) # number of patches, assumes that image dimensions are divisble by the blocksizes

  xᴸᴸᴿ = zeros(T,Wy*Wz,L,K)
  for i=1:K
    xᴸᴸᴿ[:,:,i] = im2colDistinct(x[:,:,i], (Wy,Wz))
  end
  xᴸᴸᴿ = permutedims(xᴸᴸᴿ,[1 3 2])

  # threshold singular values
  Threads.@threads for i = 1:L
    if xᴸᴸᴿ[:,:,i] == zeros(T, Wy*Wz,K)
      continue
    end
    SVDec = svdfact(xᴸᴸᴿ[:,:,i])
    proxL1!(SVDec[:S],λ)
    xᴸᴸᴿ[:,:,i] = SVDec[:U]*diagm(SVDec[:S])*SVDec[:Vt]
  end

  # reshape into image
  xᵗʰʳᵉˢʰ = zeros(T,size(x))
  for i = 1:K
    xᵗʰʳᵉˢʰ[:,:,i] = col2imDistinct( xᴸᴸᴿ[:,i,:], (Wy,Wz), (ny,nz) )
  end

  if randshift
    xᵗʰʳᵉˢʰ = circshift(xᵗʰʳᵉˢʰ, -1*shift_idx)
  end

  if !isempty(shape)
    xᵗʰʳᵉˢʰ = reshape( xᵗʰʳᵉˢʰ, prod(shape),floor( Int, length(xᵗʰʳᵉˢʰ)/prod(shape) ) )
  end

  return xᵗʰʳᵉˢʰ
end

@doc "return the value of the LLR-regularization term" ->
normLLR(reg::Regularization,x) = normLLR(x; reg.params...)

function normLLR(x::Vector; shape::NTuple=[], L=1, blockSize::Array{Int64,1}=[2; 2], randshift::Bool=true, kargs...)

  N = prod(shape)
  K = floor(Int,length(x)/(N*L))
  normᴸᴸᴿ = 0.
  for i = 1:L
    normᴸᴸᴿ +=  blockNuclearNorm(x[(i-1)*N*K+1:i*N*K], shape; blockSize=blockSize, randshift=randshift, kargs...)
  end

  return normᴸᴸᴿ
end

function blockNuclearNorm{T}(x::Vector{T}, shape::Tuple; blockSize::Array{Int64,1}=[2; 2], randshift::Bool=true, kargs...)
    x = reshape( x, tuple( shape...,floor(Int64, length(x)/prod(shape)) ) )

    Wy = blockSize[1]
    Wz = blockSize[2]

    if randshift
      srand(1234)
      shift_idx = [rand(1:Wy) rand(1:Wz) 0]
      x = circshift(x, shift_idx)
    end

    ny, nz, K = size(x)

    # reshape into patches
    L = floor(Int,ny*nz/Wy/Wz) # number of patches, assumes that image dimensions are divisble by the blocksizes

    xᴸᴸᴿ = zeros(T,Wy*Wz,L,K)
    for i=1:K
      xᴸᴸᴿ[:,:,i] = im2colDistinct(x[:,:,i], (Wy,Wz))
    end
    xᴸᴸᴿ = permutedims(xᴸᴸᴿ,[1 3 2])

    # L1-norm of singular values
    normᴸᴸᴿ = 0.
    for i = 1:L
      SVDec = svdfact(xᴸᴸᴿ[:,:,i])
      normᴸᴸᴿ += norm(SVDec[:S],1)
    end

    return normᴸᴸᴿ
end
