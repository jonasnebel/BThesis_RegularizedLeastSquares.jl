module RegularizedLeastSquares

import Base: length
using FFTW
using LinearAlgebra
import LinearAlgebra.BLAS: gemv, gemv!
import LinearAlgebra: BlasFloat, normalize!, norm, rmul!, lmul!
using LinearOperators
using SparseArrays
using ProgressMeter
using IterativeSolvers
using Random

export createLinearSolver, init, deinit, solve, linearSolverList,linearSolverListReal

abstract type AbstractLinearSolver end
const Trafo = Union{AbstractMatrix, AbstractLinearOperator, Nothing}
const FuncOrNothing = Union{Function, Nothing}

# Fallback function
setlambda(S::AbstractMatrix, λ::Real) = nothing

include("proximalMaps/ProxL1.jl")
include("proximalMaps/ProxL2.jl")
include("proximalMaps/ProxL21.jl")
include("proximalMaps/ProxLLR.jl")
# include("proximalMaps/ProxSLR.jl")
include("proximalMaps/ProxPositive.jl")
include("proximalMaps/ProxProj.jl")
include("proximalMaps/ProxTV.jl")
include("proximalMaps/ProxNuclear.jl")

include("Regularization.jl")
include("LinearOperator.jl")
include("Utils.jl")
include("Kaczmarz.jl")
include("DAX.jl")
include("CGNR.jl")
include("CG.jl")
include("Direct.jl")
include("FusedLasso.jl")
include("FISTA.jl")
include("ADMM.jl")
include("SplitBregman.jl")

"""
Return a list of all available linear solvers
"""
function linearSolverList()
  Any["kaczmarz","cgnr"] # These are those passing the tests
    #, "fusedlasso"]
end

function linearSolverListReal()
  Any["kaczmarz","cgnr","daxkaczmarz","daxconstrained"] # These are those passing the tests
    #, "fusedlasso"]
end


"""
This file contains linear solver that are commonly used in MPI
Currently implemented are
 - kaczmarz method (the default)
 - CGNR
 - A direct solver using the backslash operator


All solvers return an approximate solution to STx = u.


Function returns choosen solver.
"""
createLinearSolver(args...; regName::String = "L2", λ::Float64 = 0.0, kargs...) =
   createLinearSolver(args...; regName=[regName], λ=[λ], kargs...)

function createLinearSolver(solver::AbstractString, A, reg=nothing, reg2=nothing;
                            regName::Vector{String}} = "L2"
                            λ::Vector{Float64} = 0.0,
                            log::Bool=false, kargs...)

  log ? solverInfo = SolverInfo(;kargs...) : solverInfo=nothing

  regu = Regularization(regName, λ; kargs...)

  if solver == "kaczmarz"
    reg =  Regularization("L2",λ)
    return Kaczmarz(A, reg; kargs...)
  elseif solver == "cgnr"
    reg =  Regularization("L2",λ)
    return CGNR(A, reg; kargs...)
  elseif solver == "direct"
    return DirectSolver(A; kargs...)
  elseif solver == "daxkaczmarz"
    return DaxKaczmarz(A; kargs...)
  elseif solver == "daxconstrained"
    return DaxConstrained(A; kargs...)
  elseif solver == "pseudoinverse"
    return PseudoInverse(A; kargs...)
  elseif solver == "fusedlasso"
    return FusedLasso(A; kargs...)
  elseif solver == "fista"
    reg==nothing ? reg = Regularization(;kargs...) : nothing
    return FISTA(A, reg;kargs...)
  elseif solver == "admm"
    reg==nothing ? reg = Regularization(;kargs...) : nothing
    return ADMM(A, reg;kargs...)
  elseif solver == "splitBregman"
    reg==nothing ? reg = Regularization() : nothing
    reg2==nothing ? reg2 = Regularization() : nothing
    return SplitBregman(A,reg,reg2;kargs...)
  else
    error("Solver $solver not found.")
  end
end

end
