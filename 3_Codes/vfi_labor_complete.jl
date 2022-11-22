## BASIC CALIBRATION
β = 0.99
θ = 0.36
ϕ = 0.5    # Change the intial guess in the function euler
γz = 0.00
γn = 0.00
δ = 0.025


## CALIBRATION FOR THE STOCHASTIC PROCESS
ρ = 0.99;
μ = 0;
σ = 0.007;
Z = 4;


## COMPUTING THE STEADY STATE CAPITAL
using NLsolve
function euler(eq, x)
    h=(x[2])
    k=(x[1])
    eq[1] = k - (β*θ / ((1+γz)-β*(1-δ)))^(1/(1-θ)) *h
    eq[2] = (1-h)*((1-θ) *k^θ * h^(-θ)) - ϕ*(k^θ * h^(1-θ) +(1-δ)*k - (1+γn)*(1+γz)k)
end
S = nlsolve(euler, [0.1,0.8],ftol = :1.0e-9, method = :trust_region , autoscale = true)
kss = S.zero[1]
hss = S.zero[2]
zss = 0.0


## GRID FOR CAPITAL
K = 200 # number of gridpoints for capital
kmax = 1.25*kss
kmin = 0.25*kss
k = range(kmin, stop = kmax, length=K)


## GRID FOR LABOR
H = K
h = range(0, stop = 1, length = H);


## BUILDING THE STOCHASTIC PROCESS

##Defining the Grid for the Exogenous State Variable: Technology Shock
using Distributions
function Tauchen(ρ,σ,N,μ,m)
    # This function is used to discretize an AR(1) process following Tauchen method
    # y_(t+1) = μ(1-ρ) + ρy_(t) + ϵ
    # ϵ ∼ N (0, σ^2)
    # N represents the number of y states

    if N>1
        yN= μ + m*(σ^2/(1-ρ^2))^(1/2) # This representes the maximum value of y
        y1= μ - m*(σ^2/(1-ρ^2))^(1/2) # This representes the minimum value of y
        Δ = (yN -y1)/(N-1) # This is the distance between each y
        y = y1:Δ:yN # This is the vector of possible states

    d = Normal()

    pdfY=ones(N,N) # This preallocate memory and create the transition matrix in the followimg loop
    for i in 1:N
        pdfY[i,1]=cdf(d,(y[1] + Δ/2 -ρ*y[i] - μ*(1- ρ)) / σ);
        pdfY[i,N]=1-cdf(d,(y[N-1] +Δ/2 - ρ*y[i] - μ*(1-ρ)) / σ);
        for j in 2:N-1
            pdfY[i,j]=cdf(d,(y[j] + Δ/2 - ρ*y[i] - μ*(1-ρ) )/σ) - cdf(d,(y[j] - Δ/2 - ρ*y[i] - μ*(1-ρ)) / σ);
        end
    end
else
    y=μ
    pdfY=1
end
    return pdfY, y
end
Π, z = Tauchen(ρ,σ,Z,μ)
z = exp.(z)
########################


## COMPUTING THE VALUES FOR INVESTMENT AND CONSUMPTION
c = Array{Float64,4}(undef,K,Z,K,H)

for k0 in 1:K, k1 in 1:K
    global x = (1 + γz)*(1 + γn)*k[k1] - (1 - δ)*k[k0] # define the investment variable
    if x >=0 # in case it is positive we assigen a normal value for consumption
        for z0 in 1:Z, h0 in 1:H
            c[k0,z0,k1,h0] = k[k0]^θ *(z[z0]*h[h0])^(1-θ) + (1-δ)k[k0] - (1+γz)*(1+γn)k[k1]
        end
    else # If not, consumption is negative  and will be ruled out by Utility function
        c[k0,:,k1, :] = -ones(1, Z, 1, H)
    end
end
return c


## UTILITY FUNCTION
function log_utility(c)
    if c <=0
        u = -Inf
    else
        u = log(c)
    end
    return u
end

if ϕ>0

    u_l = log_utility.(c)

    u = Array{Float64,3}(undef,K,Z,K)
    policy_h1_index = Array{Float64,3}(undef,K,Z,K)

    for k0 in 1:K, z0 in 1:Z, k1 in 1:K
        u[k0,z0,k1], policy_h1_index[k0,z0,k1] = findmax(u_l[k0,z0,k1,:] .+ ϕ.*log.(1 .- h))
    end
else
    u =log_utility.(c[:,:,:,end])
    real_policy_h_index = H .*trunc.(Int, ones(K,Z))
end


## VALUE FUNCTION ITERATION
tol = 1e-10;
its = 0;
V = zeros(K,Z)
Vf= zeros(K,Z)
distance = 1;
maxits = 800;
policy_k_index = Array{Int64,2}(undef,K,Z)
policy_c_index = Array{Float64,2}(undef,K,Z)


while distance>tol && its<maxits
    global its = its + 1
     Vf = copy(V) # We do this to save V to compare later

     # We need to find the expected value of the value function, i.e., E[V(k,z')|z] = π(z1|z)*V(k, z1) + π(z2|z)*V(k,z2)
    EVf= (Π*Vf')'

    for k0 in 1:K, z0 in 1:Z
        V[k0,z0] , policy_k_index[k0,z0] = findmax(u[k0,z0,:] .+ β *(1+γn)* EVf[:,z0]) #Bellman Equation
    end
global distance = maximum(abs.(V-Vf))
end


## CREATING THE POLICY FUNCTION FOR CAPITAL
policy_k = Array{Float64,2}(undef,K,Z);

for j=1:Z ## Z different policy functions for l(one for each state of nature)
policy_k[:,j] = k[policy_k_index[:,j]]
end


## CREATING THE POLICY FUNCTION FOR LABOR
if  ϕ > 0
real_policy_h_index = Array{Int64,2}(undef,K,Z)
for k0 in 1:K, z0 in 1:Z
    real_policy_h_index[k0,z0] = policy_h1_index[k0,z0,policy_k_index[k0,z0]];
end
else
    real_policy_h_index = H .*trunc.(Int, ones(K,Z))
end

policy_h = Array{Float64,2}(undef,K,Z);

for j=1:Z ## Z different policy functions for l(one for each state of nature)
policy_h[:,j] = h[real_policy_h_index[:,j]]
end


## CREATING THE POLICY FUNCTION FOR CONSUMPTION
policy_c = Array{Float64,2}(undef,K,Z) ## Z different policy functions for c(one for each state of nature)
for k1 in 1:K, z0 in 1:Z
    policy_c[k1,z0] = k[k1]^(θ)*(exp(z[z0])*policy_h[k1,z0])^(1-θ)-policy_k[k1,z0]*(1+γz)*(1+γn)+(1-δ)k[k1]
end


## CREATING THE PLOTS
using Plots
plot(k,V,title="Value Function",label=["Value Function for z1" "Value Function for z2" "Value Function for z3" "Value Function for z4"])
#savefig("valuefunction.png")
plot(k,policy_c,title="Policy functions for consumption",label=["Optimal Consumption for z1" "Optimal Consumption for z2" "Optimal Consumption for z3" "Optimal Consumption for z4"])
#savefig("consumption.png")
plot(k,policy_k,title="Policy functions for capital",label=["Optimal Capital for z1" "Optimal Capital for z2" "Optimal Capital for z3" "Optimal Capital for z4"])
#savefig("capital.png")
plot(k,policy_h,title="Policy functions for labor supply",label=["Optimal labor supply for z1" "Optimal labor supply for z2" "Optimal labor supply for z3" "Optimal labor supply for z4"])
#savefig("labor.png")
