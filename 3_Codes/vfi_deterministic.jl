
##Define Parameters
α =  0.36 # Capital share in production function
β = 0.99 # Discount factor
δ = 0.025 # Depreciation rate


##Utility function
u = (ct) -> log(ct)


##Producution function
F = (kt) -> kt^(α)


##Defining the steady state
k_ss = (α/((1/β) - (1-δ)))^(1/(1-α));  # Steady State Capital Stock
y_ss = F(k_ss);                # Steady State Output
c_ss = y_ss - δ*k_ss;           # Steady State Consumption


##Defining the Grid for the Endogenous State Variable: Capital
nk = 501;                            # Number of Grid Points
kmin = 0.2*k_ss; kmax = 1.8*k_ss;    # Bounds for Grid
kg = kmin:(kmax-kmin)/(nk-1):kmax; # Equally Spaced Grid for Capital


## Build the 2-Dimensional Contemporaneous Utility Grid for the System
Ut = zeros(nk,nk)

for ii = 1:nk       # Loop Over Capital Today
    for jj = 1:nk   # Loop Over Capital Tomorrow
        k = kg[ii];     # Capital Today
        kp = kg[jj];    # Capital Tomorrow
        # Solve for Consumption at Each Point
        c = F(k) + (1-δ)*k - kp
        if (kp < 0)||(c .< 0)
            # If Tomorrow"s Capital Stock | Today"s Consumption is Negative
            Ut[ii,jj] = -9999999999;
            # Numerical Trick to ensure that the Value Function is never
            # optimised at these pointselse
        else
            Ut[ii,jj] = u(c);
            # Calculate Utility at this Point on Grid
        end
    end
end


##Value Function Iteration

#Initial Guess of the Value Function
V0 = ones(nk,1) # nk x 1 vector of initial guess

tol = 1e-10;
its = 0;
maxits = 3000; # Define the maximum number of iterations
V_1 = V0;  # The new value function I obtain after an iteration
V_0 = V0;  # the  value function from which I start in each new iteration
dif = 1;
policy_k_index = Array{Int64,1}(undef,nk);
maxits = 3000


while dif>tol && its <maxits
    global its = its + 1;
    V_0 = copy(V_1);
    for k0 in 1:nk
    V_1[k0], policy_k_index[k0] = findmax(Ut[k0,:] + β*V_0[:])
    end
global dif = maximum(abs.(V_1-V_0));
println("Iteration: $(its)")
end


##Policy function for capital
policy_k = Array{Float64,1}(undef,nk);

policy_k = kg[policy_k_index]


##Policy function for consumption
policy_c = Array{Float64,1}(undef,nk)

for k1 in 1:nk
policy_c[k1] = F(kg[k1]) + (1-δ)*kg[k1] - policy_k[k1]
end


##Plotting the results
using Plots
plot(kg,V_1,title="Value Function",label="Value Function")
plot(kg,policy_c,title="Policy functions for consumption",label="Optimal Consumption")
plot(kg,policy_k,title="Policy functions for capital",label="Optimal Capital")
