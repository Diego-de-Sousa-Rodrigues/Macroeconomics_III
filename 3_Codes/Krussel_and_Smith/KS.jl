
# Agent problem in Krussell-Smith

using Interpolations, ProgressMeter
using Distributions, Random, DataFrames, GLM


include("CRRA_utility.jl")

# Wages functions
R(K::Float64,H::Float64,z::Float64;α=α::Float64,δ=δ::Float64)= z*α*K^(α-1.0)*H^(1.0-α) + (1.0-δ)
w(K::Float64,H::Float64,z::Float64;α=α::Float64) = z*(1.0-α)*K^(α)*H^(-α)

# Law of motion functions for aggregate states
K1(K::Float64,z::Float64;b=b::Array{Float64,2},Z= Z::Array{Float64,1}) = exp(b[findfirst(Z.==z),1]+b[findfirst(Z.==z),2]*log(K))
H0(K::Float64,z::Float64;d=d::Array{Float64,2},Z = Z::Array{Float64,1}) = exp(d[findfirst(Z.==z),1]+d[findfirst(Z.==z),2]*log(K))

# Defining asset investment function:
a1star(c,a,e,k,h,z) = R(k,h,z)*a-c+w(k,h,z)*e*nstar(c,e,k,h,z)

# Optimal labor choice. Labor FOC:
nstar(c,e,k,h,z;η = η,lbar = lbar) = (e>0.0)*min(max((lbar - (1-η)/η * c/w(k,h,z)),0.0),lbar)



function ENDOGENOUSGRID_KS(A::Array{Float64,1},A1::Array{Float64,1},E::Array{Float64,1},Z::Array{Float64,1},transmat::Array{Float64,2},states::NTuple,
    K::Array{Float64,1}, H::Array{Float64,1} ,b::Array{Float64,2},d::Array{Float64,2};α=α::Float64,β = β::Float64, η=η::Float64, μ=μ::Float64,
     tol = 1e-6, lbar=lbar::Float64 ,  policy= zeros(nA,nE,nK,nH,nZ)::Array{Float64,6},update_policy=0.5::Float64,updaterule = false)

    #This function solves the agent problem using the endogenous grid method.
    #A: Individual Asset grid in t
    #A: Individual Asset grid in t+1
    #E: Individual productivity grid
    #Z: Aggregate shocks grid
    #transmat: transmat object with all the transition matrices
    #states: A tuple which each element is a pair of possible states
    #K: Aggregate capital grid
    #H: Aggregate labor grid
    #b: capital law of motion coefficients
    #d: labor law of motion coefficients


    nA::Int64 = length(A)
    nZ::Int64 = length(Z)
    nE::Int64 = length(E)
    nH::Int64 = length(H)


    itpc = LinearInterpolation((A,E,K,H,Z),policy, extrapolation_bc=Line())
    policy_c(a,e,k,h,z) = itpc(a,e,k,h,z)
    policy_n(a,e,k,h,z) = nstar(policy_c(a,e,k,h,z),e,k,h,z)
    policy_a(a,e,k,h,z) = a1star(policy_c(a,e,k,h,z),a,e,k,h,z)


    # Loop:
    policy1= copy(policy) # Store updated values
    prog = ProgressUnknown("Iterations:")
    iteration ::Int64 = 0
    distance::Float64 = 1.0
    dist1 = policy1.-policy #

    while distance > tol
        # Return the new policygrid
    @inbounds   innerforloop!(policy1,policy_c,policy_n,b,d;
            A=A,E=E,Z=Z,K=K,H = H,lbar = lbar,A1=A1)

        # Check convergence
        distance = maximum(abs.(policy1-policy))

        # Error if no convergence
        if distance == NaN || distance == Inf
            error("Agent Problem did not converge")
        end

        # Based ond Carrol
        dist = copy(dist1)
        dist1 = policy1.-policy
        if iteration >1
            φ = dist1./dist
            φ[dist1.<tol] .= 0.0
            φ[φ.>1.0] .=0.5
            φ[0.9.<φ.<=1.0] .= 0.9
        end
        if iteration > 4 && updaterule
            policy = (policy1.- φ.*policy)./(1.0.-φ)
        else
            policy = update_policy*policy1 + (1.0-update_policy)*policy1
        end

        # Update the policy functions:
        itpc = LinearInterpolation((A,E,K,H,Z),policy, extrapolation_bc=Line())
        ProgressMeter.next!(prog; showvalues = [(:Distance, distance)])
        iteration +=1
        if iteration == 500 || iteration > 1200
            update_policy = rand()
        elseif iteration >10000
            break
        end

    end
    ProgressMeter.finish!(prog)
    println("Agent problem finished with a distance of $(distance)")
    return  policy
end


function innerforloop!(policy1::Array{Float64,5}, policy_c::Function,policy_n,b,d;
    A1=A1,E=E,Z=Z,K=K,H = H,lbar = lbar,A=A)
    # Returns the updated policy grid given the policies functions
    A0 = copy(A1)
    for ki = 1:length(K)
        k=K[ki]
         for (zi,z) = enumerate(Z),(hi,h) = enumerate(H),(ei,e) = enumerate(E)
            for (ai,a1) = enumerate(A1) # a1 is assets tommorow
            # Find the level of assets and consumption today that generates a1 given the policy functions
                policy1[ai,ei,ki,hi,zi],A0[ai] = EE(a1;e=e,z=z,K=k,H=h,b=b,d=d,η=η,policy_c = policy_c)
            end

            # sort the asset today (needed for the Interpolation function)
            ordem = sortperm(A0)
            # interpolate consumption today as a function of today's:
            itpc0 = LinearInterpolation(A0[ordem],policy1[ordem,ei,ki,hi,zi],extrapolation_bc=Line())

            # Update the grid:
            for ai = 1:length(A)
                if A0[1]<=A1[ai]
                    policy1[ai,ei,ki,hi,zi] = itpc0(A1[ai])

                else


                    policy1[ai,ei,ki,hi,zi] = η*(R(k,h,z)*A1[ai]- A1[1] + w(k,h,z)*e*lbar)
                end
            end
        end
    end
    return policy1
end

function EE(a1;e=E[e]::Float64,
    z=Z[z]::Float64,K=K[k]::Float64,H=H[h]::Float64,states=states::NTuple{4,Array{Float64,1}},
    b=b::Array{Float64,2},d=d::Array{Float64,2},η=η::Float64,Z = Z,E = E,policy_c=policy_c)
    # a1 is the asset level tommorow
    # Finds assets today as a function of assets tomorrow using the Euler equations
    i::Int64 = findfirstn(states,[z,e]) # find the current state index
    k1::Float64 = K1(K,states[i][1];b=b) # Aggregate states tommorow given today
    h1::Float64 = H0(k1,states[i][1];d=d)
    RHS1::Float64 = 0.0 # Find the RHS of the consumption FOC uct'= βE[R uct1 ']
    for e1=1:nE, z1 = 1:nZ # for all possible states tommorow
        j::Int64 = findfirstn(states,[Z[z1],E[e1]]) # find the tommorow state index
        c1::Float64 = policy_c(a1,E[e1],k1,h1,Z[z1]) # find consumption in t+1 given policy function
        a2::Float64 = a1star(c1,a1,E[e1],k1,h1,Z[z1]) # find assets in t+2 given policy function
        n1::Float64 = nstar(c1,E[e1],k1,h1,Z[z1]) # find labor in t+1 given policy function
        l1::Float64 = lbar - n1 #leisure
        RHS1 += β * transmat[i,j]*R(k1,h1,Z[z1])*uc(c1,l1) # The RHS for the state j given i
    end

    # Find the level of consumption today that generates a1 given the policy functions
    if e > 0.0
        c = (RHS1/η * ((1-η)/(η*e*w(K,H,z)))^(-(1-μ)*(1-η)))^(-1/μ)
    else
        c = (RHS1/η *lbar^(-(1-μ)*(1-η)))^(1/(η*(1-μ)-1))
    end
    # Find the consitent asset level for today (endogenous grid)
    a = (c+a1-e*w(K,H,z)*nstar(c,e,K,H,z))/R(K,H,z)

    return c,a
end


function KrusselSmithENDOGENOUS(A::Array{Float64,1},A1::Array{Float64,1},
    E::Array{Float64,1},Z::Array{Float64,1},tmat::TransitionMatrix,states::NTuple{4,Array{Float64,1}},
    K::Array{Float64,1},H::Array{Float64,1},  b::Array{Float64,2},d::Array{Float64,2};
    α = α::Float64,β = β::Float64, η = η::Float64, μ=μ::Float64, tol= 1e-6::Float64,
    update_policy=0.5::Float64,updateb= 0.3::Float64,N::Int64=5000,T::Int64=11000,
    discard::Int64=1000,seed::Int64= 2803,lbar=lbar::Float64,updaterule = false)
    #This performs KS algorithm
    #A: Individual Asset grid in t!
    #A1: Individual Asset grid in t+1
    #E: Individual productivity grid
    #Z: Aggregate shocks grid
    #tmat: transmat object with all the transition matrices
    #states: A tuple which each element is a pair of possible states
    #K: Aggregate capital grid
    #H: Aggregate labor grid
    #b: capital law of motion coefficients
    #d: labor law of motion coefficients


    #RETURN
    #b: Updated parameter for aggregate capital law of motions
    #d: Updated parameter for aggregate labor law of motions
    #nsim: NxT matrix with simulated labor path for each agent n
    #asim: NxT matrix with simulated assets path for each agent n
    #Ksim: T vector with simulated aggregate  capital
    #Hsim: T vector with simulated aggregate  Labor ,
    #policygrid: Grid with agents policy functions
    #K: new updated grid for aggregate capita (not used for now),
    #R2b,R2d: R-squared of b and d regressions
    #zsimd: T vector with simulated aggregate shocks
    #esim: NxT matrix with idyosincratic employment shock for each agent

    # Getting lengths
    nA::Int64 = length(A)
    nZ::Int64 = length(Z)
    nE::Int64 = length(E)
    nH::Int64 = length(H)
    nK::Int64 = length(K)

    println("Starting Krusell Smith. We are using $(nA) gridpoints for assets and")
    println("a sample of N=$(N), T=$(T). Go somewhere else, this will take a while.")

    transmat::Array{Float64,2} = tmat.P # Getting the transition matrix for the agent

    d=d::Array{Float64,2}


    zi_shock,epsi_shock = generate_shocks(KSParameter(); z_shock_size = T, population = N)

    zsim = fill(1.01,T)
    zsim[zi_shock.==2].=0.99
    zsimd::Array{Float64,1} = zsim[discard+1:end] #Discarded simulated values for z


    esim = fill(1.0,N,T)
    esim[epsi_shock' .== 2] .= 0.0

    meanUgood = 1-mean(esim[:,zsim.==Z[2]])
    meanUbad = 1-mean(esim[:,zsim.==Z[1]])

    println("Unemployment in bad state is $(meanUbad) and $(meanUgood) in good states.")

    # Predefining variables
    asim::Array{Float64,2} = rand(K[1]:0.1:K[end],N,T) # The initial assets will generate aggregate assets in the grid
    Ksim::Array{Float64,1} = ones(T)
    Hsim::Array{Float64,1} = ones(T)
    nsim::Array{Float64,2} = ones(N,T)
    R2d::Array{Float64,1} = ones(2)
    R2b::Array{Float64,1} = ones(2)

    # First guessess for Policy
    policygrid::Array{Float64,5} =  ones(nA,nE,nK,nH,nZ)
    for (zi,z) = enumerate(Z),(hi,h) = enumerate(H),(ki,k) = enumerate(K),(ei,e)=enumerate(E),(ai,a1)=enumerate(A)
        policygrid[ai,ei,ki,hi,zi] = 0.9*a1
    end
    itpc = LinearInterpolation((A,E,K,H,Z),policygrid, extrapolation_bc=Line())
    multiple100::Int64 = 0

    policy_c(a,e,k,h,z) = itpc(a,e,k,h,z)
    policy_n(a,e,k,h,z) = nstar(policy_c(a,e,k,h,z),e,k,h,z)
    policy_a(a,e,k,h,z) = a1star(policy_c(a,e,k,h,z),a,e,k,h,z)

    # Loop
    dist::Float64 = 1.0
    iteration::Int64 = 0
    b1::Array{Float64,2} = copy(b)
    d1::Array{Float64,2} = copy(d) #to store updated values for b and d

    Ht::Float64 = 1.0

    while (dist>tol)

        println("Solving the agent problem")
        # Solve the agent problem:
        policygrid = ENDOGENOUSGRID_KS(A,A1,E,Z,transmat,states,K, H,b,d;policy= policygrid,update_policy=update_policy,tol = tol,updaterule = updaterule)
        itpc = LinearInterpolation((A,E,K,H,Z),policygrid, extrapolation_bc=Line())
        println("Agent Problem solved!")


        loading = Progress(T, 1,"Simulating the economy.", 30)
        #Simulating the economy

        for t=1:T
            Ksim[t] = mean(asim[:,t]) # Aggregate capital is the mean of the capital decided yesterday
            # First guess for aggregate labor:
            Ht = H0(Ksim[t],zsim[t];d=d)
            # Find aggregate labor that clears the market:
            internaldist = 10.0
            its = 0
                while internaldist>1e-6 && its < 500
                    nsim[:,t] .= policy_n.(asim[:,t],esim[:,t],Ksim[t],Ht,zsim[t]) # Store each agent labor decision
                    Hsim[t] = mean(nsim[:,t])
                    internaldist = abs(Hsim[t] - Ht)
                    Ht = Hsim[t]
                    its+=1
                end

                if t<T
                    asim[:,t+1] .= policy_a.(asim[:,t],esim[:,t],Ksim[t],Ht,zsim[t]) # Store each agent asset decision
                end
            next!(loading) # Loading bar
        end


        println("Economy simulated,run the regression")
        # Regreession
        for i=1:nZ # for each state
            datad = DataFrame(Xd = log.(Ksim[discard+1:end][zsimd.==Z[i]]),
            Yd = log.(Hsim[discard+1:end][zsimd.==Z[i]])) # Take log of capital and labor and dataframe
            olsd = lm(@formula(Yd ~ Xd), datad) # Regress
            d1[i,:] = coef(olsd) # Coefficients
            R2d[i] = r2(olsd) # R2

            datab = DataFrame(Xb = log.(Ksim[discard+1:end-1][zsimd[1:end-1].==Z[i]]),
            Yb = log.(Ksim[discard+2:end][zsimd[1:end-1].==Z[i]])) # Take log of capital and capital tomorrow and dataframe
            olsb = lm(@formula(Yb ~ Xb), datab) # Regress
            b1[i,:] = coef(olsb)# Coefficients
            R2b[i] = r2(olsb) # R2

        end


        # Convergence
        dist = maximum(vcat(abs.(b.-b1),abs.(d.-d1)))

        # Law of motions with a damping parameter
        b = updateb.*b1 .+ (1-updateb).*b
        d = updateb.*d1 .+ (1-updateb).*d

        iteration += 1
        println("In iteration $(iteration), law distance is $(dist)")
        println("b = $(b) and")
        println("d = $(d)")
        println("Aggregate labor mean is $(mean(Hsim))")
        println("Aggregate Capital mean is $(mean(Ksim))")
        println("Aggregate Capital R2 is $(R2b)")
        println("Aggregate Labor R2 is $(R2d)")
    end
    
    println("Krussell Smith done!")
    return b, d,  nsim, asim, Ksim, Hsim,policygrid,K,R2b,R2d,zsimd,esim
end
