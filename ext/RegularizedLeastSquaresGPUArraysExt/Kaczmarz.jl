function RegularizedLeastSquares.iterate_row_index(solver::Kaczmarz, state::RegularizedLeastSquares.KaczmarzState{T, vecT}, A, row, index) where {T, vecT <: AbstractGPUArray}
  state.τl = RegularizedLeastSquares.dot_with_matrix_row(A,state.x,row)
  @allowscalar state.αl = solver.denom[index]*(state.u[row]-state.τl-state.ɛw*state.vl[row])
  RegularizedLeastSquares.kaczmarz_update!(A,state.x,row,state.αl)
  @allowscalar state.vl[row] += state.αl*state.ɛw
end

function RegularizedLeastSquares.kaczmarz_update!(A::matT, x::vecT, row, beta) where {T, matT <: AbstractGPUArray{T}, vecT <: AbstractGPUVector{T}}
  x[:] .=  x .+ beta * conj.(view(A, row, :))
end

function RegularizedLeastSquares.kaczmarz_update!(B::Transpose{T, S}, x::vecT, row, beta) where {T, S <: AbstractGPUArray{T}, vecT <: AbstractGPUVector{T}}
  A = B.parent
  x[:] .=  x .+ beta * conj.(view(A, :, row))
end


function RegularizedLeastSquares.kaczmarz_update!(prod::ProdOp{Tc, WeightingOp{T, vecT}}, x, k, beta) where {T, Tc<:Union{T, Complex{T}}, vecT <: AbstractGPUVector{T}}
  weight = @allowscalar prod.A.weights[k]
  RegularizedLeastSquares.kaczmarz_update!(prod.B, x, k, weight*beta) # only for real weights
end

function RegularizedLeastSquares.dot_with_matrix_row(state_τl::vecT, A::matT, x::matT, k::Int64) where {T, vecT <: AbstractVector{T}, matT <: AbstractGPUArray{T}}
  # state_τl .= vec(sum(x .* view(A, k, :), dims = 1))
  # state_τl .= vec(sum(collect(Broadcast.instantiate(Broadcast.broadcasted(*, view(A, k, :), x))), dims = 1))
  state_τl .= sum(Broadcast.instantiate(Broadcast.broadcasted(*, view(A, k, :), x)), dims = 1, init = 0.0)'
end
