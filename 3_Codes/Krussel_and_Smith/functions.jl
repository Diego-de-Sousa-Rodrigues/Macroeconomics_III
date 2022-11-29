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


struct TransitionMatrix
    # We used the Hori(2018) functions available in the quantecon
    P::Matrix{Float64}       # 4x4
    Pz::Matrix{Float64}      # 2x2 aggregate shock
    Peps_gg::Matrix{Float64} # 2x2 idiosyncratic shock conditional Expansion/Expansion
    Peps_bb::Matrix{Float64} # 2x2 idiosyncratic shock conditional Recession/Recession
    Peps_gb::Matrix{Float64} # 2x2 idiosyncratic shock conditional Expansion/Recession
    Peps_bg::Matrix{Float64} # 2x2 idiosyncratic shock conditional Recession/Expansion
end


function create_transition_matrix(;ug::AbstractFloat=0.04, # unemployment rate in expansions
    ub::AbstractFloat=0.1, # unemployment rate in recessions
    zg_ave_dur::Real=8, # average duration of both an expansion and a recession
    zb_ave_dur::Real=8, # average duration of both an expansion and a recession
    ug_ave_dur::Real=1.5, # average duration of unemployment in expansion
    ub_ave_dur::Real=2.5, # average duration of unemployment in recessions
    puu_rel_gb2bb::Real=1.25,
    puu_rel_bg2gg::Real=0.75)
    # Compute the transition matrices
    # We used the Hori(2018) functions available in the quantecon
    # probability of remaining in expansion state
    pgg = 1-1/zg_ave_dur
    # probability of remaining in recession state
    pbb = 1-1/zb_ave_dur
    # probability of changing from good to bads
    pgb = 1-pgg
    # probability of changing from bad to good
    pbg = 1-pbb

    # prob. of 0 to 0 cond. on g to g
    p00_gg = 1-1/ug_ave_dur
    # prob. of 0 to 0 cond. on b to b
    p00_bb = 1-1/ub_ave_dur
    # prob. of 0 to 1 cond. on g to g
    p01_gg = 1-p00_gg
    # prob. of 0 to 1 cond. on b to b
    p01_bb = 1-p00_bb

    # prob. of 0 to 0 cond. on g to b
    p00_gb=puu_rel_gb2bb*p00_bb
    # prob. of 0 to 0 cond. on b to g
    p00_bg=puu_rel_bg2gg*p00_gg
    # prob. of 0 to 1 cond. on g to b
    p01_gb=1-p00_gb
    # prob. of 0 to 1 cond. on b to g
    p01_bg=1-p00_bg

    # prob. of 1 to 0 cond. on  g to g
    p10_gg=(ug - ug*p00_gg)/(1-ug)
    # prob. of 1 to 0 cond. on b to b
    p10_bb=(ub - ub*p00_bb)/(1-ub)
    # prob. of 1 to 0 cond. on g to b
    p10_gb=(ub - ug*p00_gb)/(1-ug)
    # prob. of 1 to 0 cond on b to g
    p10_bg=(ug - ub*p00_bg)/(1-ub)
    # prob. of 1 to 1 cond. on  g to g
    p11_gg= 1-p10_gg
    # prob. of 1 to 1 cond. on b to b
    p11_bb= 1-p10_bb
    # prob. of 1 to 1 cond. on g to b
    p11_gb= 1-p10_gb
    # prob. of 1 to 1 cond on b to g
    p11_bg= 1-p10_bg

                #b0         b1        g0           g1
    P = [pbb*p00_bb pbb*p01_bb pbg*p00_bg pbg*p01_bg;
         pbb*p10_bb pbb*p11_bb pbg*p10_bg pbg*p11_bg;
         pgb*p00_gb pgb*p01_gb pgg*p00_gg pgg*p01_gg;
         pgb*p10_gb pgb*p11_gb pgg*p10_gg pgg*p11_gg]

    Pz=[pbb pbg;
        pgb pgg]

    Peps_gg=[p00_gg p01_gg;
            p10_gg p11_gg]
    Peps_bb=[p00_bb p01_bb;
            p10_bb p11_bb]
    Peps_gb=[p00_gb p01_gb;
            p10_gb p11_gb]
    Peps_bg=[p00_bg p01_bg;
            p10_bg p11_bg]

    transmat=TransitionMatrix(P, Pz, Peps_gg, Peps_bb, Peps_gb, Peps_bg)
    return transmat
end


function idioshocks(zsim::Array{Float64,1},transmat::TransitionMatrix;N=N::Int64,
    Z=Z::Array{Float64,1},E=E::Array{Float64,1}, e0 = E[1]::Float64,ug=ug::Float64,ub=ub::Float64)
    # This function finds employment shocks consistent with aggregate shocks distribution
    T::Int64 = length(zsim)
    nE::Int64 = length(E)
    nZ::Int64 =length(E)
    esim::Array{Float64,2} = fill(0.0,N,T) # This is the states path
    p::Float64 = 0.0
    u::Int64=0
    r::Array{Float64,2} = rand(N,T) # Random draws

    #First period:
    for n=1:N
        if zsim[1]==Z[1]
            if r[n,1]>ub
                esim[n,1] = E[2]
            else
                esim[n,1] = E[1]
            end
        else
            if r[n,1]>ug
                esim[n,1] = E[2]
            else
                esim[n,1] = E[1]
            end
        end
    end

    # Simulating the economy
    for n = 1:N
      for t=2:T
            e0 = findfirstn(E,esim[t-1])
            z0 = findfirstn(Z,esim[t-1])
            z1 = findfirstn(Z,esim[t])
            if z0==Z[1] && z1==Z[1] # conditional on bad to bad and e0
                if r[n,t] <= transmat.Peps_bb[e0,1]
                    esim[n,t]=E[1]
                else
                    esim[n,t]=E[2]
                end
            elseif z0==Z[1] && z1==Z[2] # conditional on bad to good and e0
                if r[n,t] <= transmat.Peps_bg[e0,1]
                    esim[n,t]=E[1]
                else
                    esim[n,t]=E[2]
                end
            elseif z0==Z[2] && z1==Z[1] # conditional on good to bad and e0
                if r[n,t] <= ransmat.Peps_gb[e0,1]
                    esim[n,t]=E[1]
                else
                    esim[n,t]=E[2]
                end
            else # conditional on good to good and e0
                if r[n,t] <= transmat.Peps_gg[e0,1]
                    esim[n,t]=E[1]
                else
                    esim[n,t]=E[2]
                end
            end


        end
    end
    return esim
end


using Interpolations # to use interpolation
using Random, LinearAlgebra
using QuantEcon  # to use `gridmake`, `<:AbstractUtility`
using Optim      # to use minimization routine to maximize RHS of bellman equation
using GLM        # to regress
using JLD2       # to save the result
using ProgressMeter # to show progress of iterations
using Parameters # to use type with keyword arguments


struct TransitionMatrix
    P::Matrix{Float64}       # 4x4
    Pz::Matrix{Float64}      # 2x2 aggregate shock
    Peps_gg::Matrix{Float64} # 2x2 idiosyncratic shock conditional on good to good
    Peps_bb::Matrix{Float64} # 2x2 idiosyncratic shock conditional on bad to bad
    Peps_gb::Matrix{Float64} # 2x2 idiosyncratic shock conditional on good to bad
    Peps_bg::Matrix{Float64} # 2x2 idiosyncratic shock conditional on bad to good
end


function create_transition_matrix(ug::Real, ub::Real,
        zg_ave_dur::Real, zb_ave_dur::Real,
        ug_ave_dur::Real, ub_ave_dur::Real,
        puu_rel_gb2bb::Real, puu_rel_bg2gg::Real)

    # probability of remaining in good state
    pgg = 1-1/zg_ave_dur
    # probability of remaining in bad state
    pbb = 1-1/zb_ave_dur
    # probability of changing from g to b
    pgb = 1-pgg
    # probability of changing from b to g
    pbg = 1-pbb

    # prob. of 0 to 0 cond. on g to g
    p00_gg = 1-1/ug_ave_dur
    # prob. of 0 to 0 cond. on b to b
    p00_bb = 1-1/ub_ave_dur
    # prob. of 0 to 1 cond. on g to g
    p01_gg = 1-p00_gg
    # prob. of 0 to 1 cond. on b to b
    p01_bb = 1-p00_bb

    # prob. of 0 to 0 cond. on g to b
    p00_gb=puu_rel_gb2bb*p00_bb
    # prob. of 0 to 0 cond. on b to g
    p00_bg=puu_rel_bg2gg*p00_gg
    # prob. of 0 to 1 cond. on g to b
    p01_gb=1-p00_gb
    # prob. of 0 to 1 cond. on b to g
    p01_bg=1-p00_bg

    # prob. of 1 to 0 cond. on  g to g
    p10_gg=(ug - ug*p00_gg)/(1-ug)
    # prob. of 1 to 0 cond. on b to b
    p10_bb=(ub - ub*p00_bb)/(1-ub)
    # prob. of 1 to 0 cond. on g to b
    p10_gb=(ub - ug*p00_gb)/(1-ug)
    # prob. of 1 to 0 cond on b to g
    p10_bg=(ug - ub*p00_bg)/(1-ub)
    # prob. of 1 to 1 cond. on  g to g
    p11_gg= 1-p10_gg
    # prob. of 1 to 1 cond. on b to b
    p11_bb= 1-p10_bb
    # prob. of 1 to 1 cond. on g to b
    p11_gb= 1-p10_gb
    # prob. of 1 to 1 cond on b to g
    p11_bg= 1-p10_bg

    #   (g1)         (b1)        (g0)       (b0)
    P=[pgg*p11_gg pgb*p11_gb pgg*p10_gg pgb*p10_gb;
       pbg*p11_bg pbb*p11_bb pbg*p10_bg pbb*p10_bb;
       pgg*p01_gg pgb*p01_gb pgg*p00_gg pgb*p00_gb;
       pbg*p01_bg pbb*p01_bb pbg*p00_bg pbb*p00_bb]
    Pz=[pgg pgb;
        pbg pbb]
    Peps_gg=[p11_gg p10_gg
             p01_gg p00_gg]
    Peps_bb=[p11_bb p10_bb
             p01_bb p00_bb]
    Peps_gb=[p11_gb p10_gb
             p01_gb p00_gb]
    Peps_bg=[p11_bg p10_bg
             p01_bg p00_bg]
    transmat=TransitionMatrix(P, Pz, Peps_gg, Peps_bb, Peps_gb, Peps_bg)
    return transmat
end


function KSParameter(;
            beta::AbstractFloat=0.99,
            alpha::AbstractFloat=0.36,
            delta::Real=0.025,
            theta::Real=1,
            k_min::Real=0,
            k_max::Real=1000,
            k_size::Integer=100,
            K_min::Real=30,
            K_max::Real=50,
            K_size::Integer=4,
            z_min::Real=0.99,
            z_max::Real=1.01,
            z_size::Integer=2,
            eps_min::Real=0.0,
            eps_max::Real=1.0,
            eps_size::Integer=2,
            ug::AbstractFloat=0.04,
            ub::AbstractFloat=0.1,
            zg_ave_dur::Real=8,
            zb_ave_dur::Real=8,
            ug_ave_dur::Real=1.5,
            ub_ave_dur::Real=2.5,
            puu_rel_gb2bb::Real=1.25,
            puu_rel_bg2gg::Real=0.75,
            mu::Real=0,
            degree::Real=7)
    if theta == 1
        u = LogUtility()
    else
        u = CRRAUtility(theta)
    end
    l_bar=1/(1-ub)
    # individual capital grid
    k_grid=
        (range(0, stop=k_size-1, length=k_size)/(k_size-1)).^degree*(k_max-k_min).+k_min
    k_grid[1] = k_min; k_grid[end] = k_max; # adjust numerical error
    # aggregate capital grid
    K_grid=range(K_min, stop=K_max, length=K_size)
    # aggregate technology shock
    z_grid=range(z_max, stop=z_min, length=z_size)
    # idiosyncratic employment shock grid
    eps_grid=range(eps_max, stop=eps_min, length=eps_size)
    s_grid=gridmake(z_grid, eps_grid)               # shock grid
    # collection of transition matrices
    transmat=create_transition_matrix(ug,ub,
        zg_ave_dur,zb_ave_dur,
        ug_ave_dur,ub_ave_dur,
        puu_rel_gb2bb,puu_rel_bg2gg)

    ksp=(u=u, beta=beta, alpha=alpha, delta=delta, theta=theta,
         l_bar=l_bar, k_min=k_min, k_max=k_max, k_grid=k_grid,
         K_min=K_min, K_max=K_max, K_grid=K_grid, z_grid=z_grid,
         eps_grid=eps_grid, s_grid=s_grid, k_size=k_size, K_size=K_size,
         z_size=z_size, eps_size=eps_size, s_size=z_size*eps_size,
         ug=ug, ub=ub, transmat=transmat, mu=mu)

    return ksp
end


function generate_shocks(ksp::NamedTuple;
                         z_shock_size::Integer = 1100,
                         population::Integer = 10000)

    # unpack parameters
    Peps_gg = ksp.transmat.Peps_gg
    Peps_bg = ksp.transmat.Peps_bg
    Peps_gb = ksp.transmat.Peps_gb
    Peps_bb = ksp.transmat.Peps_bb

    # draw aggregate shock
    zi_shock = simulate(MarkovChain(ksp.transmat.Pz), z_shock_size)

    ### Let's draw individual shock ###
    epsi_shock = Array{Int}(undef, z_shock_size, population) # preallocation

    # first period
    rand_draw=rand(population)
    # recall: index 1 of eps is employed, index 2 of eps is unemployed
    if zi_shock[1] == 1 # if good
        epsi_shock[1, :] .= (rand_draw .< ksp.ug) .+ 1 # if draw is higher, become employed
    elseif zi_shock[1] == 2 # if bad
        epsi_shock[1, :] .= (rand_draw .< ksp.ub) .+ 1 # if draw is higher, become employed
    else
        error("the value of z_shocks[1] (=$(z_shocks[1])) is strange")
    end

    # from second period ...
    for t = 2:z_shock_size
        draw_eps_shock!(Val(zi_shock[t]), Val(zi_shock[t-1]),
                        view(epsi_shock, t, :), epsi_shock[t-1, :], ksp.transmat)
    end

    # adjustment
    for t=1:z_shock_size
        n_e = count(epsi_shock[t,:].==1) # count number of employed
        empl_rate_ideal = ifelse(zi_shock[t] == 1, 1.0-ksp.ug, 1.0-ksp.ub)
        gap = round(Int, empl_rate_ideal*population) - n_e
        if gap > 0
            become_employed_i = rand(findall(2 .== epsi_shock[t,:]), gap)
            epsi_shock[t, become_employed_i] .= 1
        elseif gap < 0
            become_unemployed_i = rand(findall(1 .== epsi_shock[t, :]), -gap)
            epsi_shock[t,become_unemployed_i] .= 2
        end
    end

    return zi_shock, epsi_shock
end


function draw_eps_shock!(epsi_shocks,
                         epsi_shock_before,
                         Peps::AbstractMatrix)
    # loop over entire population
    for i=1:length(epsi_shocks)
        rand_draw=rand()
        epsi_shocks[i]=ifelse(epsi_shock_before[i] == 1,
                              (Peps[1, 1] < rand_draw)+1,  # if employed before
                              (Peps[2, 1] < rand_draw)+1)  # if unemployed before
    end
    return nothing
end


draw_eps_shock!(zi::Val{1}, zi_lag::Val{1}, epsi,
                epsi_lag::AbstractVector, transmat::TransitionMatrix) =
    draw_eps_shock!(epsi, epsi_lag, transmat.Peps_gg)
draw_eps_shock!(zi::Val{1}, zi_lag::Val{2}, epsi,
                epsi_lag::AbstractVector, transmat) =
    draw_eps_shock!(epsi, epsi_lag, transmat.Peps_bg)
draw_eps_shock!(zi::Val{2}, zi_lag::Val{1}, epsi,
                epsi_lag::AbstractVector, transmat) =
    draw_eps_shock!(epsi, epsi_lag, transmat.Peps_gb)
draw_eps_shock!(zi::Val{2}, zi_lag::Val{2}, epsi,
                epsi_lag::AbstractVector, transmat) =
    draw_eps_shock!(epsi, epsi_lag, transmat.Peps_bb)
