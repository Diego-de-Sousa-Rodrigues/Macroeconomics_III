# Solves the agent problem using the Reiter's method

using Interpolations, ProgressMeter,LinearAlgebra,SparseArrays
using IterativeSolvers, ForwardDiff
include("CRRA_utility.jl")

# Returns
R(z::Real,k::Real,l::Real;α = α::Real,δ=δ::Real,A=A::Real)=  α*A*z*k^(α-1)*l^(1-α) + 1-δ
w(z::Real,k::Real,l::Real;α = α::Real,A=A::Real)=  (1-α)*A*z*k^α*l^(-α)

# Law of motion functions for Aggregate states
K1(K::Real,z::Real;b=b::AbstractArray,Z= Z::AbstractArray) = exp(b[findfirst(Z.==z),1]+b[findfirst(Z.==z),2]*log(K))
H0(K::Real,z::Real;d=d::AbstractArray,Z = Z::AbstractArray) = exp(d[findfirst(Z.==z),1]+d[findfirst(Z.==z),2]*log(K))


function weight(a1::Real,a::Real,e::Real,pol::Real,grid::AbstractArray)

    a1i::Real = findfirstn(grid,a1)
    ai::Real = findfirstn(grid,a)

    if pol <= grid[1] && a1i==1
        return 1.0

    elseif a1i<length(grid) && a1<= pol <= grid[a1i+1]
        return (grid[a1i+1] - pol)/(grid[a1i+1] - a1)
    elseif a1i>1 && grid[a1i-1]<=pol<= a1
        return (pol-grid[a1i-1])/(a1-grid[a1i-1])
    elseif pol >= grid[end] && a1i== length(grid)
        return 1.0
    else
        return 0.0
    end
end


function λ1(a1::Real,e1::Real,λ::AbstractArray,R::Real,w::Real,policy_a::Function;E=E::AbstractArray,grid = Ad::AbstractArray,pdfE=pdfE::AbstractArray,E_size=E_size::Real)
    lambda::Real = 0.0
    e1i::Real = findfirstn(E,e1)
    for ei =1:E_size, ai =1:Ad_size
       lambda += pdfE[ei,e1i] * weight(a1,grid[ai],E[ei],policy_a(grid[ai],E[ei],R=R,w=w),grid) * λ[ai,ei]
    end
    return lambda
end


function weightmatrix(policy_a::Function,R::Real,w::Real;Ad_size=Ad_size::Real, E_size = E_size::Real,
    E=E::AbstractArray,grid = Ad::AbstractArray,pdfE=pdfE::AbstractArray)
    weights::AbstractArray = ones(Ad_size,E_size,Ad_size,E_size)
  for ai=1:Ad_size
        for ei=1:E_size,e1i=1:E_size,a1i = 1:Ad_size
            weights[a1i,e1i,ai,ei] = pdfE[ei,e1i] * weight(grid[a1i],grid[ai],E[ei],policy_a(grid[ai],E[ei],R=R,w=w),grid)
        end
    end
    return reshape(weights,E_size*Ad_size,E_size*Ad_size)
end


function invariantλ(λ::AbstractArray,weight::AbstractArray;Ad=Ad::AbstractArray,E=E::AbstractArray,tol = 1e-7::Real)
dist::Real = 1.0
lambda2=copy(λ)
    while dist > tol
        lambda2 = weight * λ
        dist = maximum(abs.(λ-lambda2))
        λ = copy(lambda2)
    end

    return λ
end


function findSS(Ap::AbstractArray,Ad::AbstractArray,E::AbstractArray,E_size::Real,
    Ad_size::Real,pdfE::AbstractArray;tol = 1e-6::Real,update = 0.8::Real,R0=1/β*0.8::Real,
    w0=1.0::Real,α=α::Real,β = β::Real,A=A::Real,η = η::Real)

    # Steady state level for Z
    zss = exp(0.0)

    # Initial L
    K0::Real = 50.0
    L0::Real = 50.0
    dist::Real = 1.0
    policygrid::AbstractArray = ones(length(Ap),E_size)
    itpc = LinearInterpolation((Ap,E),policygrid, extrapolation_bc=Line())
    lambda::AbstractArray = fill(1/(E_size*Ad_size),Ad_size*E_size)
    policy_c(a,e) = itpc(a,e)
    policy_n(a,e;w=w0) = nstar(policy_c(a,e),e,w)
    policy_a(a,e;w=w0,R=R0) = a1star(policy_c(a,e),a,e,w,R)
    λ::AbstractArray = fill(1/(Ad_size*E_size),Ad_size,E_size)
    W::AbstractArray = (weightmatrix(policy_a,R0,w0))
    prog = ProgressUnknown("Looking for the SS, iterations:")
    while dist > tol

        # Policy function given returns (R and w):
        policygrid = ENDOGENOUSGRID(R0,w0,Ap,Ap,E,pdfE;policy= policygrid)

        itpc = LinearInterpolation((Ap,E),policygrid, extrapolation_bc=Line())

        # Invariant distribution implied by policy functions:
        W = (weightmatrix(policy_a,R0,w0))
        lambda = fill(1/(E_size*Ad_size),Ad_size*E_size)
        powm!(W, lambda, maxiter = 100000,tol = 1e-15)
        λ = reshape(lambda,(Ad_size,E_size))./sum(lambda)

        # Aggregate Capital and Labor
        K0 = sum(policy_a.(Ad,E[1]).*λ[:,1] .+ policy_a.(Ad,E[2]).*λ[:,2])
        L0 = sum(policy_n.(Ad,E[1]).*λ[:,1] .+ policy_n.(Ad,E[2]).*λ[:,2])

        R1::Real = min((α*A*zss*K0^(α-1)*L0^(1-α)+1-δ),1/β)
        w1::Real = ((1-α)*A*zss*K0^α*L0^(-α))
        dist = max(abs(R0-R1),(η!=1.0)* abs(w0-w1))

        # Bisection method
        R0::Real = update * R1 + (1-update) * R0
        if η!=1.0
            w0::Real = update * w1 + (1-update) * w0
        else
            w0=copy(w1)
        end
        ProgressMeter.next!(prog; showvalues = [(:Distance, dist),(:AggregateK,K0),(:AggregateL,L0),(:R,R0),(:w,w0)])
    end
    ProgressMeter.finish!(prog)
    println("Aggregates converged with a distance of $(dist)")

return policygrid,K0,L0,R0,w0,λ
end
