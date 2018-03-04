using Metropolis
using Random

srand(999)

const N = 20  # lattice size
const a = 1.0  # lattice spacing
const ω = 1.0
const m = 1.0

const Ncf = 1000
const Ncor = 20
const Nthermalize = 10Ncor

const ϵ = 1.4

using Metropolis: update!


uniform(a::Real, b::Real) = (b-a)*rand() + a
uniform() = uniform(-ϵ, ϵ)


function S(x::Vector{<:Real}, j::Integer)
    j₊ = mod1(j+1, N)
    j₋ = mod1(j-1, N)
    a*m*ω^2*x[j]^2/2 + (m/a)*x[j]*(x[j] - x[j₊] - x[j₋])
end
S(x::Vector{<:Real}) = sum(S(x, i) for i ∈ eachindex(x))

Metropolis.fluctuate(x::Vector{<:Real}, j::Integer) = (x[j] += uniform())


function createΓ(n::Integer, t::Integer)
    x::Vector -> x[t]*x[mod1(t+n, N)]
end


function G(I::MetropolisIntegrator, n::Integer, x::Vector{<:Real})
    mean(vev(I, S, createΓ(n, t), x) for t ∈ 1:N)
end

function initx()
    x = zeros(Float64, N)
    thermalize!(Nthermalize, S, x)
    x
end


function ΔE(n::Integer)
    I = MetropolisIntegrator(Ncf, Ncor, Nthermalize)
    x = initx()
    Gn = G(I, n, x)
    x = initx()
    Gn1 = G(I, n+1, x)
    log(Gn/Gn1)/a
end


# should return ω
E1 = ΔE(1)

