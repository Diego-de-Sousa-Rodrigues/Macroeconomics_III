### Fall, 2021
### Diego de Sousa Rodrigues

using Plots
using ForwardDiff
using NLsolve


##Define Parameters
θ =  0.36 # Capital share in production function
β = 0.99  # Discount factor
δ = 0.025 # Depreciation rate
ψ = 0.5   # Disutility from labor
nk = 200  # Number of grid points for capital
nh = 200  # Number of grid points for labor


##Define Parameters for the Stochastic Shock
ρ = 0.99  # Persistence of the technology shock
σ = 0.007 # Standard deviation of the technology shock
nz = 4    # Number of stochastic shocks (grid points for the stochastic shock)
μ = 0     # AR(1) constant term
m = 3     # Scaling parameter


##Computing the steady state values for capital and labor

# EULER EQUATION
function euler(eq, x)
    h=(x[2])
    k=(x[1])
    eq[1] = k - (β*θ / (1-β*(1-δ)))^(1/(1-θ)) *h
    eq[2] = (1-h)*((1-θ) *k^θ * h^(-θ)) - ψ *(k^θ * h^(1-θ) +(1-δ)*k - k)
end
S = nlsolve(euler, [0.1,0.8],ftol = :1.0e-9, method = :trust_region , autoscale = true)
kss = S.zero[1]
hss = S.zero[2]
zss = 1.0

# GRID FOR CAPITAL
kmax = 1.25*kss
kmin = 0.25*kss
k = range(kmin, stop = kmax, length=nk)

# GRID FOR LABOR
h = range(0, stop = 1, length = nh);


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

Π, z = Tauchen(ρ,σ,nz,μ,m)
z = exp.(z)


##Computing the values for investment and consumption
x = Array{Float64,2}(undef,nk,nk) # Define the investment variable
for k0 in 1:nk, k1 in 1:nk
    x[k0,k1] = k[k1] - (1-δ)*k[k0];
end

c = Array{Float64,4}(undef,nk,nk,nz,nh) # Define the consumption variable
for k0 in 1:nk, k1 in 1:nk, z0 in 1:nz, h0 in 1:nh
    c[k0,k1,z0,h0] = (k[k0])^θ*(z[z0]*h[h0])^(1-θ)+(1-δ)*k[k0]-k[k1];
end

# Now we will penalyze the consumption, such that if the investment is negative the consumption will be negative in a way
# that will never be optimal to choose those points

for k0 in 1:nk, k1 in 1:nk, z0 in 1:nz, h0 in 1:nh
    if x[k0,k1]<0;
        c[k0,k1,z0,h0]=-1;
    else
    end
end


##Building the utility functions
function log_consumption(c)
    if c <=0
        uc = -Inf
    else
        uc = log(c)
    end
    return uc # This defines the utility for consumption only
end

function log_labor(h)
        uh = log(1-h)
    return uh # This defines the utility for labor only
end


##Finding the optimal value for labor

if ψ>0
    uc = log_consumption.(c)
    uh = log_labor.(h)

    u = Array{Float64,3}(undef,nk,nk,nz) # This defines the utility where we have the optimal h
    policy_h_index = Array{Float64,3}(undef,nk,nk,nz)

    for k0 in 1:nk, k1 in 1:nk, z0 in 1:nz
        u[k0,k1,z0], policy_h_index[k0,k1,z0] = findmax(uc[k0,k1,z0,:] .+ ψ.*uh[:])
    end

else
    u =log_consumption.(c[:,:,:,end])
    real_policy_h_index = nh .*trunc.(Int, ones(nk,nz))
end


##VALUE FUNCTION ITERATION

tol = 1e-10;
its = 0;
maxits = 3000; # Define the maximum number of iterations
V0 = zeros(nk,nz)
V1= zeros(nk,nz)
distance = 1;
policy_k_index = Array{Int64,2}(undef,nk,nz)
policy_c_index = Array{Float64,2}(undef,nk,nz)


while distance>tol && its<maxits
    global its = its + 1
     V0 = copy(V1) # We do this to save V0 to compare later
     # We need to find the expected value of the value function, i.e., E[V(k,z')|z] = π(z1|z)*V(k, z1) + π(z2|z)*V(k,z2)
    EVf= (Π*V0')'
    for k0 in 1:nk, z0 in 1:nz
        V1[k0,z0] , policy_k_index[k0,z0] = findmax(u[k0,:,z0] .+ β * EVf[:,z0]) # Bellman Equation
    end
global distance = maximum(abs.(V1-V0))
println("Iteration: $(its)")
end


## CREATING THE POLICY FUNCTION FOR CAPITAL
policy_k = Array{Float64,2}(undef,nk,nz);

for j=1:nz # nz different policy functions for capital (one for each state of nature)
    policy_k[:,j] = k[policy_k_index[:,j]]
end


## CREATING THE POLICY FUNCTION FOR LABOR
if  ψ > 0
    real_policy_h_index = Array{Int64,2}(undef,nk,nz)
    for k0 in 1:nk, z0 in 1:nz
        real_policy_h_index[k0,z0] = policy_h_index[k0,policy_k_index[k0,z0],z0];
    end
else
    real_policy_h_index = nh .*trunc.(Int, ones(nk,nz))
end

policy_h = Array{Float64,2}(undef,nk,nz);

for j=1:nz # nz different policy functions for labor (one for each state of nature)
    policy_h[:,j] = h[real_policy_h_index[:,j]]
end


## CREATING THE POLICY FUNCTION FOR CONSUMPTION
policy_c = Array{Float64,2}(undef,nk,nz) # nz different policy functions for consumption (one for each state of nature)
for k1 in 1:nk, z0 in 1:nz
    policy_c[k1,z0] = k[k1]^(θ)*((z[z0])*policy_h[k1,z0])^(1-θ)-policy_k[k1,z0]+(1-δ)k[k1]
end


## CREATING THE PLOTS
using Plots
plot(k,V1,title="Value Function",label=["Value Function for z1" "Value Function for z2" "Value Function for z3" "Value Function for z4"])
#savefig("valuefunction.png")
plot(k,policy_c,title="Policy functions for consumption",label=["Optimal Consumption for z1" "Optimal Consumption for z2" "Optimal Consumption for z3" "Optimal Consumption for z4"])
#savefig("consumption.png")
plot(k,policy_k,title="Policy functions for capital",label=["Optimal Capital for z1" "Optimal Capital for z2" "Optimal Capital for z3" "Optimal Capital for z4"])
#savefig("capital.png")
plot(k,policy_h,title="Policy functions for labor supply",label=["Optimal labor supply for z1" "Optimal labor supply for z2" "Optimal labor supply for z3" "Optimal labor supply for z4"])
#savefig("labor.png")
