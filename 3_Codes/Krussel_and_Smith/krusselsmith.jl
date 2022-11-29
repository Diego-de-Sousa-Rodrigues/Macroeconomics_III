include("functions.jl")
include("KS.jl")

#Parameters from KS
const α = 0.36
const β = 0.99
const δ = 0.025

const η = 1.0/2.9
const μ = 1.0

const N = 5000 # Agents
const T = 11000 # Simulation periods
const discard = 1000 # Periods to discard
const lbar = 1.0 # Endowment of time

const nA = 30 # Gridpoints of assets
const nK = 6 # Gridpoints of capital
const nH = 6 # Gridpoints of labor

amin = 0.0
amax = 350.0

ug=0.04 # unemployment rate in good times/expansions
ub=0.1 # unemployment rate in bad times/recessions
zg_ave_dur=8 # average duration of good period
zb_ave_dur=8 # average duration of bad period
ug_ave_dur=1.5 # average duration of unemployment period in good times
ub_ave_dur=2.5 # average duration of unemployment period in bad times
puu_rel_gb2bb=1.25 # imposed conditions from KS
puu_rel_bg2gg=0.75 # imposed conditions

# Productivity shocks/ Aggregate shocks
const Z = [0.99,1.01]
const nZ=length(Z)

# Employment shocks/ Idiosyncratic shocks
const E = [0.0,1.0]
const nE = length(E)


const tmat = create_transition_matrix()
const transmat = tmat.P

const states = ([Z[1],E[1]],[Z[1],E[2]],[Z[2],E[1]],[Z[2],E[2]])

# Guessing Law of motions parameters
b = [0.114 0.953;0.123 0.951] # taken from KS
d = [-0.592 -0.255;-0.544 -0.252]  # taken from KS

# Asset grid:
factor =4.0
const A =(range(0, stop=nA-1, length=nA)/(nA-1)).^factor * amax
const A1 =A
K = range(5,stop = 20.0, length = nK).^1

Hk= ones(nK,2) .*lbar
for j=1:2,i =1:nK-1
    Hk[i,j] = H0(K[i],Z[j])
end


H = range(0.001,stop = lbar,length = nH).^1.0

b, d,  nsim, asim, Ksim, Hsim,policygrid,K,R2b,R2d,zsim,esim = KrusselSmithENDOGENOUS(A,A1,E,Z,tmat,states,K,H,b,d;
N=N,T=T,discard = discard, update_policy=.75,updateb= .4, updaterule = true,seed =1234)


using JLD2, FileIO
@save "allvariables.jld2"
@save "variables.jld2" b d  nsim asim Ksim Hsim policygrid K R2b R2d zsim esim discard Z T


# @load "allvariables.jld2"
# @load "variables.jld2"


using Plots
#Aggregate Capital vs capital:
plot((Ksim[discard+1:end-1][zsim[1:end-1].==Z[1]]) ,(Ksim[discard+2:end][zsim[1:end-1].==Z[1]]),
xlabel = "\$K_t\$",ylabel = "\$K_{t+1}\$",linestyle = :dot, label = "Bad State",legend = :bottomright)
savefig("figure1")

plot!((Ksim[discard+1:end-1][zsim[1:end-1].==Z[2]]) ,(Ksim[discard+2:end][zsim[1:end-1].==Z[2]]),
linestyle = :dot, label = "Good State")
savefig("figure2")


plot!(11.7:0.1:12.7 ,11.7:0.1:12.7, label = "45 degrees")
savefig("figure3")

#Aggregate labor vs capital:
plot((Ksim[discard+1:end][zsim.==Z[1]]) ,(Hsim[discard+1:end][zsim.==Z[1]]),
xlabel = "\$K_t\$",ylabel = "\$H_t\$",linestyle = :dot, label = "Bad State")
savefig("figure4")

plot!((Ksim[discard+1:end][zsim.==Z[2]]) ,(Hsim[discard+1:end][zsim.==Z[2]]),
linestyle = :dot,label = "Good State")
savefig("figure5")


# Law of motion functions for aggregate states

K1(K::Float64,z::Float64;b=b::Array{Float64,2},Z= Z::Array{Float64,1}) = exp(b[findfirst(Z.==z),1]+b[findfirst(Z.==z),2]*log(K))
H0(K::Float64,z::Float64;d=d::Array{Float64,2},Z = Z::Array{Float64,1}) = exp(d[findfirst(Z.==z),1]+d[findfirst(Z.==z),2]*log(K))

Ksimimplied = fill(Ksim[discard+1],T-discard)
Hsimimplied = fill(Hsim[discard+1],T-discard)
for t = 1:T-discard
    if t<T-discard
        Ksimimplied[t+1] = K1(Ksimimplied[t],zsim[t])
    end
    Hsimimplied[t] = H0(Ksimimplied[t],zsim[t])
end

plot(Ksimimplied,ylabel="\$K_t\$",xlabel="Time",label = "Aggregate Capital",linestyle = :dot)
plot!(Ksim[discard+1:end],label = "Implied by law of motion")
savefig("figure6.png")

plot(Hsimimplied,ylabel="\$H_t\$",xlabel="Time",label = "Aggregate Labor",linestyle = :dot)
plot!(Hsim[discard+1:end],label = "Implied by law of motion")
savefig("figure7")

# Creating the summary statistics
# Call again the parameters

Ysimimplied = zsim.*Ksimimplied.^(α).*Hsimimplied.^(1-α)

Ysim = zsim.*Ksim[discard+1:end].^(α).*Hsim[discard+1:end].^(1-α)

plot(Ysim,ylabel="\$Y_t\$",xlabel="Time",label = "Output",linestyle = :dot)
savefig("figure8")

Rsimimplied = zsim.*α.*Ksimimplied.^(α-1.0).*Hsimimplied.^(1.0-α) .+ (1.0-δ)
wsimimplied = zsim.*(1.0-α).*Ksimimplied.^(α).*Hsimimplied.^(-α)

Rsim = zsim.*α.*Ksim[discard+1:end].^(α-1.0).*Hsim[discard+1:end].^(1.0-α) .+ (1.0-δ)
wsim = zsim.*(1.0-α).*Ksim[discard+1:end].^(α).*Hsim[discard+1:end].^(-α)

plot(Rsim, ylabel="\$R_t\$",xlabel = "Time",linestyle = :dot, label = "\$R_t\$",legend = :bottomright)
plot!(wsim, ylabel="\$w_t\$ and \$R_t\$",xlabel = "Time",linestyle = :dot, label = "\$w_t\$",legend = :bottomright)
savefig("figure9")

# Building the investment and the consumption variale

Isim = Ksim[discard+1:end] .- (1.0-δ).*Ksim[discard:end-1]

Csim = Ysim .- Isim

plot(Isim, ylabel="\$I_t\$",xlabel = "Time",linestyle = :dot, label = "\$I_t\$",legend = :bottomright)
savefig("figure10")

plot(Csim, ylabel="\$C_t\$",xlabel = "Time",linestyle = :dot, label = "\$C_t\$",legend = :bottomright)
savefig("figure11")

using Statistics

Y = hcat(Ysim, Csim, Isim)

C = cor(Y)

S = std(Y, dims=1)

M = mean(Y, dims=1)
