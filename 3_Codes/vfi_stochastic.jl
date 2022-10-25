
##Define Parameters
α =  0.36 # Capital share in production function
β = 0.99 # Discount factor
δ = 0.025 # Depreciation rate
ρ = 0.99 # Persistence of the technology shock
σ = 0.007 # Standard deviation of the technology shock


##Utility function
u = (ct) -> log(ct)


##Producution function
F = (kt,zt) -> zt*kt^(α)


##Defining the steady state
z_ss = 1;                            # Steady State Technology Shock
k_ss = (α/((1/β) - (1-δ)))^(1/(1-α));  # Steady State Capital Stock
y_ss = F(k_ss,z_ss);                # Steady State Output
c_ss = y_ss - δ*k_ss;           # Steady State Consumption


##Defining the Grid for the Endogenous State Variable: Capital
nk = 101;                            # Number of Grid Points
kmin = 0.2*k_ss; kmax = 1.8*k_ss;    # Bounds for Grid
kg = kmin:(kmax-kmin)/(nk-1):kmax; # Equally Spaced Grid for Capital


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

m = 3; # Scaling parameter
nz = 9; # Number of grid points
μ=0;

Π, zg = Tauchen(ρ,σ,nz,μ,m)
zg = exp.(zg)


## Build the 3-Dimensional Contemporaneous Utility Grid for the System
Ut = zeros(nk,nk,nz)

for kk = 1:nz           # Loop Over Technology Today
    for ii = 1:nk       # Loop Over Capital Today
        for jj = 1:nk   # Loop Over Capital Tomorrow
            z = zg[kk];     # Technology Today
            k = kg[ii];     # Capital Today
            kp = kg[jj];    # Capital Tomorrow
            # Solve for Consumption at Each Point
            c = F(k,z) + (1-δ)*k - kp
            if (kp < 0)||(c .< 0)
                # If Tomorrow"s Capital Stock | Today"s Consumption is Negative
                Ut[ii,jj,kk] = -9999999999;
                # Numerical Trick to ensure that the Value Function is never
                # optimised at these pointselse
            else()
                Ut[ii,jj,kk] = u(c);
                # Calculate Utility at this Point on Grid
            end
        end
    end
end


##Value Function Iteration

#Initial Guess of the Value Function
V0 = kron(zg',ones(nk,1)) # zg' is 1 x nz & ones(nk,1) is nk x 1
# V0 is nk x nz

#Calculate the Guess of the Expected Value Function
EVf = V0 * Π'


tol = 1e-10;
its = 0;
maxits = 3000; # Define the maximum number of iterations
V_1 = V0;  # The new value function I obtain after an iteration
V_0 = V0;  # the  value function from which I start in each new iteration
dif = 1;
policy_k_index = Array{Int64,2}(undef,nk,nz);
maxits = 3000


while dif>tol && its <maxits
    global its = its + 1;
    V_0 = copy(V_1);
    EVf = (Π*V_0')'
    for k0 in 1:nk, z0 in 1:nz
    V_1[k0,z0], policy_k_index[k0,z0] = findmax(Ut[k0,:,z0] + β*EVf[:,z0])
    end
global dif = maximum(abs.(V_1-V_0));
println("Iteration: $(its)")
end


##Policy function for capital
policy_k = Array{Float64,2}(undef,nk,nz);

for j=1:nz
policy_k[:,j] = kg[policy_k_index[:,j]] # nz different policy functions for k(one for each state of nature)
end


##Policy function for consumption
policy_c = Array{Float64,2}(undef,nk,nz) # nz different policy functions for c(one for each state of nature)

for k1 in 1:nk, z0 in 1:nz
policy_c[k1,z0] = F(kg[k1],zg[z0]) + (1-δ)*kg[k1] - policy_k[k1,z0]
end


##Plotting the results
using Plots
plot(kg,V_1,title="Value Function",label=["Value Function for z1" "Value Function for z2" "Value Function for z3" "Value Function for z4" "Value Function for z5" "Value Function for z6" "Value Function for z7" "Value Function for z8" "Value Function for z9"])
plot(kg,policy_c,title="Policy functions for consumption",label=["Optimal Consumption for z1" "Optimal Consumption for z2" "Optimal Consumption for z3" "Optimal Consumption for z4" "Optimal Consumption for z5" "Optimal Consumption for z6" "Optimal Consumption for z7" "Optimal Consumption for z8" "Optimal Consumption for z9"])
plot(kg,policy_k,title="Policy functions for capital",label=["Optimal Capital for z1" "Optimal Capital for z2" "Optimal Capital for z3" "Optimal Capital for z4" "Optimal Capital for z5" "Optimal Capital for z6" "Optimal Capital for z7" "Optimal Capital for z8" "Optimal Capital for z9"])
