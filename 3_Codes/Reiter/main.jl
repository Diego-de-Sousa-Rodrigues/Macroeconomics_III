# This is the main file


include("Reiter.jl")
include("Householdproblem.jl")

using Distributions

function Tauchen(ρ,σ,Y,μ = 0,m = 3)
    # This function is going to be used to discretize an AR(1) space
    if Y>1
        ybar = μ/(1-ρ)
        ymax= ybar + m*(σ^2/(1-ρ^2))^(1/2)
        ymin= ybar - m*(σ^2/(1-ρ^2))^(1/2)

        Δ = (ymax-ymin)/(Y-1)
        y=ymin:Δ:ymax

        d=Normal()

        pdfY=ones(Y,Y)
        for i in 1:Y
            pdfY[i,1]=cdf(d,(y[1] + Δ/2 -ρ*y[i]) / σ^0.5);
            pdfY[i,Y]=1-cdf(d,(y[Y] -Δ/2 - ρ*y[i]) / σ^0.5);
            for j in 2:Y-1
                pdfY[i,j]=cdf(d,(y[j] + Δ/2 - ρ*y[i])/σ^0.5) - cdf(d,(y[j] - Δ/2 - ρ*y[i]) / σ^0.5);
            end
        end
    else
        y=μ
        pdfY=1
    end

    return pdfY, y
end


function simMC(S,pdf,T,s0)
    #This function simulates a Markov chain
    #S possible states
    #pdF transition matrix of the states
    #n number of state variables controlled by pdf
    #T simulation length
    #s0 initial state
    nS::Int64 = length(S)
    ssim = fill(s0, T) #This is the states path
    r = rand(T)
    s::Int64=1

    # Simulating the economy
    for t=2:T
        s = findfirstn(S,ssim[t-1])
        ps = pdf[s,1]
        for i=1:nS
            if r[t]<ps
                ssim[t]=S[i]
                break
            else
                ps+=pdf[s,i+1]
            end
        end
    end
    return ssim
end


function findfirstn(A,b)
    s=0
    nA=length(A)
    for i=1:nA
        if A[i] == b
            s = i
            break
        end
    end
        return s
end


# Parameters
const α = 0.36
const β = 0.99
const δ = 0.025
const η = 1.0/2.9
const μ = 1.0
const A= 1.0
const lbar = 1.0

const Ap_size = 30 # Grid points for policy rules
const Ad_size = 150 # Assets gridpoints for distribution
amin = 0.0
amax = 50.0

factor =7
const Ap =(range(0.0, stop=Ap_size-1, length=Ap_size)/(Ap_size-1)).^factor * amax
const Ad = range(0.0, stop=amax, length=Ad_size).^1.0

update = 0.2


# Aggregate shocks
ρ = 0.9
σ = 0.0001
Z1(z;ρ=ρ,σ =σ) = exp(ρ * log(z) + σ * randn())
zss=1.0


# Idiosyncratic shocks
E_size = 5 # Number of states for e
σe = 0.3
ρe = 0.6
pdfE,E = Tauchen(ρe,σe,E_size,1-ρe)    #E comes from Tauchen method


# Guesses for the Returns
w0 = 2.3739766754889238
R0=1.010012230693079


# Find the steady state

policygridss,Kss,Lss,Rss,wss,λss = findSS(Ap,Ad,E,E_size,Ad_size,pdfE;tol = 1e-6,update = .1,R0 = R0,w0=w0)


#SKIP THIS PART AND GO DIRECTLY TO THE PLOTS#
################################################################################################
################################################################################################
################################################################################################
# Law of motions

function λ1(λ,K,L,Z;R=R,w=w,policy_a=policy_a)
    weight = weightmatrix(policy_a,R,w)
    lambda = weight * λ
return lambda
end

function K1(λ,K,L,Z;R=R,w=w,policy_a=policy_a,Ad=Ad)
    Knew = sum(policy_a.(Ad,E[1]).*λ[:,1] .+ policy_a.(Ad,E[2]).*λ[:,2])
    return Knew
end

function L1(λ,K,L,Z;policy_n=policy_n,Ad=Ad)
    L0 = sum(policy_n.(Ad,E[1]).*λ[:,1] .+ policy_n.(Ad,E[2]).*λ[:,2])
    return L0
end

Z1(z;ρ=ρ,σ =σ) = exp(ρ * log(z) + σ * randn())
function pol(λ,K,L,Z;pss=policygridss,λss=λss,Kss=Kss,Lss=Lss,Zss=zss,pλ=pλ,pK=pK,pL=pL,pZ=pZ)
    # pλ is a matrix ApxAd
    poli =exp.(log.(pss) .+ pK*(log(K).-log(Kss)) .+ pL*(log(L).-log(Lss)) .+ pZ*(log(Z).-log(Zss)) .+ pλ*(log(λ).-log(λss)))

    return poli
end

function lR(λ,K,L,Z;Rss=Rss,λss=λss,Kss=Kss,Lss=Lss,Zss=zss,pλ=pλ,pK=pK,pL=pL,pZ=pZ)
    # Rλ is a matrix 1xAd
    poli =exp.(log.(Rss) .+ RK*(log(K).-log(Kss)) .+ RL*(log(L).-log(Lss)) .+ RZ*(log(Z).-log(Zss)))

    return poli
end

function lw(λ,K,L,Z;wss=wss,λss=λss,Kss=Kss,Lss=Lss,Zss=zss,pλ=pλ,pK=pK,pL=pL,pZ=pZ)
    # Rλ is a matrix 1xAd
    poli =exp.(log.(wss) .+ wK*(log(K).-log(Kss)) .+ wL*(log(L).-log(Lss)) .+ wZ*(log(Z).-log(Zss)))

    return poli
end


function system(λ,K,L,Z;pdfE=pdfE,E=E,policygridss=policygridss,Kss=Kss,Lss=Lss,
    Rss=Rss,wss=wss,λss=λss,ρ=ρ,σ =σ,α=α,β=β,δ=δ,Ap=Ap)
    R = R(λ,K,L,Z)
    w=  w(λ,K,L,Z)
    policygrid = pol(λ,K,L,Z)
    itpc = LinearInterpolation((Ap,E),pol(λ,K,L,Z), extrapolation_bc=Line())
    policy_c(a,e) = itpc(a,e)
    policy_n(a,e) = nstar(policy_c(a,e),e,w)
    policy_a(a,e;w=w,R=R) = a1star(policy_c(a,e),a,e,w,R)

    lambdanew = λ1(λ,K,L,Z;R=R,w=w,policy_a=policy_a)
    Knew = K1(λ,K,L,Z;R=R,w=w,policy_a=policy_a,Ad=Ad)
    Lnew = L1(λ,K,L,Z;policy_n=policy_n,Ad=Ad)
    Znew = Z1(z;ρ=ρ,σ =σ)

    return Rp,wp,policygrid,lambda1,Knew,Lnew,Znew
end

function polgrid(states;Ap=Ap,Ad=Ad,E=E,pdfE=pdfE,E_size=E_size)
    K,L,Z = states[end-2:end]
    λ = reshape(states[1:end-3],length(Ad),E_size)
    Rp = R(Z,K,L)
    wp = w(Z,K,L)
    pgrid = ENDOGENOUSGRID(Rp,wp,Ap,Ap,E,pdfE)
return pgrid[:]
end

using ForwardDiff
ForwardDiff.jacobian(polgrid, vcat(λss[:],Kss,Lss,zss))

states = vcat(λss[:],Kss,Lss,zss)
bla = polgrid(vcat(λss[:],Kss,Lss,zss))

# Returns
Rp = R(Z,K,L)
wp = w(Z,K,L)

# Policy functions

policygrid = ENDOGENOUSGRID(Rp,wp,Ap,A1p,E,pdfE)
itpc = LinearInterpolation((Ap,E),policygrid, extrapolation_bc=Line())
policy_c(a,e) = itpc(a,e)
policy_n(a,e) = nstar(policy_c(a,e),e,w)
policy_a(a,e;w=wp,R=Rp) = a1star(policy_c(a,e),a,e,w,R)

lambda1 = λ1(λ,K,L,Z;R=Rp,w=wp,policy_a=policy_a)
Knew = K1(λ,K,L,Z;R=Rp,w=wp,policy_a=policy_a)
Lnew = L1(λ,K,L,Z;policy_n=policy_n,Ad=Ad)
Znew = Z1(z;ρ=ρ,σ =σ)
################################################################################################
################################################################################################
################################################################################################


# Solving the Ayagari Model

using Plots
itpc = LinearInterpolation((Ap,E),policygridss, extrapolation_bc=Line())
    policy_c(a,e) = itpc(a,e)
    policy_n(a,e;w=w0) = nstar(policy_c(a,e),e,w)
    policy_a(a,e;w=w0,R=R0) = a1star(policy_c(a,e),a,e,w,R)

plot(Ap,policy_a.(Ap,E[3]),legend=:bottomright,label = "e= $(E[3])",title="Assets policy")
plot!(Ap,policy_a.(Ap,E[5]),label = "e= $(E[5])")
plot!(Ap,Ap,label = "45")
savefig("figure1")

plot(Ap,policy_c.(Ap,E[3]),legend=:bottomright,label = "e= $(E[3])",title="Consumption policy")
plot!(Ap,policy_c.(Ap,E[5]),label = "e= $(E[5])")
savefig("figure2")

plot(Ap,policy_n.(Ap,E[3]),legend=:bottomright,label = "e= $(E[3])",title="Labor policy")
plot!(Ap,policy_n.(Ap,E[5]),label = "e= $(E[5])")
savefig("figure3")

plot(Ad,sum(λss,dims=2),title="Assets Steady State distribution", label = "Assets")
savefig("figure4")
