__precompile__(true)

module Metropolis
using Random

srand(999) # for testing

export MetropolisIntegrator, vev

abstract type AbstractIntegrator end


struct MetropolisIntegrator{T} <: AbstractIntegrator
    Ncf::Int
    Ncor::Int
    Nthermalize::Int
    Γhistory::Vector{T}
    # TODO probably will make sense to cache action
end


function MetropolisIntegrator(Ncf::Integer, Ncor::Integer, Nthermalize::Integer)
    MetropolisIntegrator{Float64}(Ncf, Ncor, Nthermalize,
                                  Float64[NaN for i ∈ 1:Ncf])
end


function fluctuate end

# decide how to update
function update(ξnew::Ξ, ξold::Ξ, Snew::St, Sold::St) where {Ξ,St<:Real}
    δS = Snew - Sold
    exp(-δS) < rand() ? ξold : ξnew
end

# TODO this is terrible, figure out how to avoid all the allocating
# update a single lattice site
function update!(S::Function, x::Any, idx::Any)
    ξold = x[idx]
    Sold = S(x, idx)
    ξnew = fluctuate(x, idx)
    x[idx] = ξnew
    Snew = S(x, idx)
    x[idx] = update(ξnew, ξold, Snew, Sold)
    Snew
end
# update the entire lattice
function update!(S::Function, x::Any)
    for i ∈ eachindex(x)
        update!(S, x, i)
    end
end
# update the entire lattice n times
update!(n::Integer, S::Function, x) = for i ∈ 1:n update!(S,x) end


thermalize!(N::Integer, S::Function, x::Any) = update!(N, S, x)
function thermalize!(I::MetropolisIntegrator, S::Function, x::Any)
    thermalize!(I.Nthermalize, S, x)
end
export thermalize!

# NOTE: for now one should thermalize outside of vev

vev(I::MetropolisIntegrator) = mean(I.Γhistory)
function vev(I::MetropolisIntegrator, S::Function, Γ::Function, x::Any)
    for j ∈ 1:I.Ncf
        I.Γhistory[j] = Γ(x)
        update!(I.Ncor, S, x)
    end
    vev(I)
end


function vev(S::Function, Γ::Function, x::Any, ::Type{T};
             Ncf::Integer, Ncor::Integer, Nthermalize::Integer,
            ) where T  # return type of Γ
    I = MetropolisIntegrator{T}(Ncf, Ncor, Nthermalize, zeros(T, Ncf))
    vev(I, S, Γ, x)
end
function vev(S::Function, Γ::Function, x::Any;
             Ncf::Integer, Ncor::Integer, Nthermalize::Integer)
    vev(s, Γ, x, Float64, Ncf=Ncf, Ncor=Ncor, Nthermalize=Nthermalize)
end


end # module
