# An abstract type for fixed-point acceleration methods
# Fixed point problem x = g(x) with residual f(x) = x - g(x)
abstract type AbstractAccelerator{T <: Real}  end

mutable struct AndersonAccelerator{T} <: AbstractAccelerator{T}
  is_type1::Bool
  mem::Int64
  dim::Int64
  iter::Int64
  x_last::AbstractVector{T}
  g_last::AbstractVector{T}
  f::AbstractVector{T}
  f_last::AbstractVector{T}
  eta::AbstractVector{T}
  F::AbstractMatrix{T}
  X::AbstractMatrix{T}
  G::AbstractMatrix{T}
  M::AbstractMatrix{T}

  function AndersonAccelerator{T}() where {T <: Real}
    new(true, 0, 0, 0, zeros(T, 1), zeros(T, 1), zeros(T, 1),  zeros(T, 1), zeros(T, 1), zeros(T, 1, 1), zeros(T, 1, 1), zeros(T, 1, 1), zeros(T, 1, 1))
  end

  function AndersonAccelerator{T}(dim::Int64, mem::Int64 = 10, is_type1::Bool = true) where {T <: Real}
    mem <= 0 && throw(DomainError(mem, "Memory has to be a positive integer."))
    dim <= 0 && throw(DomainError(dim, "Dimension has to be a positive integer."))
    new(is_type1, mem, dim, 0, zeros(T,dim), zeros(T, dim), zeros(T, dim),  zeros(T, dim), zeros(T, mem), zeros(T, dim, mem), zeros(T, dim, mem), zeros(T, dim, mem), zeros(T, mem, mem))
  end

end

function update_history!(aa::AbstractAccelerator{T}, x::AbstractVector{T}, g::AbstractVector{T}) where {T <: Real}
  j = (aa.iter % aa.mem) + 1

  # compute residual
  @. aa.f = x - g

  # fill memory with deltas
  @. aa.X[:, j] = x - aa.x_last
  @. aa.G[:, j] = g - aa.g_last
  @. aa.F[:, j] = aa.f - aa.f_last

  if aa.is_type1
    aa.M[:, :] = aa.X' * aa.F
  else
    aa.M[:, :] = aa.F' * aa.F
  end

  # set previous values for next iteration
  @. aa.x_last = x
  @. aa.g_last = g
  @. aa.f_last = aa.f

  aa.iter += 1
end

# BLAS gesv! wrapper with error handling
# solve A X = B and for X and save result in B
function _gesv!(A, B)
  try
    LinearAlgebra.LAPACK.gesv!(A, B)
    return 1
   catch
     return -1
  end
end

function accelerate!(g::AbstractVector{T}, x::AbstractVector{T}, aa::AndersonAccelerator{T}) where {T <: Real}
  l = min(aa.iter, aa.mem)
  l == 1 && return true

  if l < aa.mem
    eta = view(aa.eta, 1:l)
    X = view(aa.X, :, 1:l)
    M = view(aa.M, 1:l, 1:l)
    G = view(aa.G, :, 1:l)
  else
    eta = aa.eta
    X = aa.X
    M = aa.M
    G = aa.G
  end

  if aa.is_type1
    eta[:] = X' * aa.f
  else
    aa.eta[:] = aa.F' * aa.f
  end
  # aa.eta = aa.M  \ (X' * f) (type1)
  info = _gesv!(M, eta)

  if (info < 0 || norm(aa.eta, 2) > 1e4)
    @warn("Acceleration failed at aa.iter: $(aa.iter)")
    return false
  else
     g[:] = g - G * eta
    return true
  end
end


struct EmptyAccelerator{T} <: AbstractAccelerator{T} end

function update_history!(ea::EmptyAccelerator{<: Real}, x::AbstractVector{T}, g::AbstractVector{T}) where {T <: Real}
  return nothing
end

function accelerate!(g::AbstractVector{T}, x::AbstractVector{T}, aa::EmptyAccelerator{T}) where {T <: Real}
  return true
end
