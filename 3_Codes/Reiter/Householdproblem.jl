# The method here is the same as in the Krusell and Smith
# Endogenous grid point

using Interpolations, ProgressMeter,LinearAlgebra

a1star(c::Real,a::Real,e::Real,w::Real,R::Real) = R*a-c+(w*e*nstar(c,e,w))

nstar(c::Real,e::Real,w::Real;η = η::Real,lbar = lbar::Real) = (e>0)*min(max((lbar - (1-η)/η * c/(w)),0.0),lbar)


function ENDOGENOUSGRID(R::Real,w::Real,A::AbstractArray,A1::AbstractArray,E::AbstractArray,
    pdfE::AbstractArray;α=α::Real,β = β::Real, η=η::Real, μ=μ::Real,
     tol = 1e-6::Real, lbar=lbar::Real ,  policy= ones(Ap_size,E_size)::AbstractArray,update_policy=0.5::Real,updaterule = false)

    itpc = LinearInterpolation((A,E),policy, extrapolation_bc=Line())
    policy_c(a,e) = itpc(a,e)
    policy_n(a,e) = nstar(policy_c(a,e),e,w)
    policy_a(a,e;w=w,R=R) = a1star(policy_c(a,e),a,e,w,R)


    policy1::AbstractArray= copy(policy)
    iteration ::Real = 0
    distance::Real = 1.0
    dist1::AbstractArray = policy1.-policy

    while distance > tol

    @inbounds   innerforloop!(policy1,policy_c,policy_n,w,R;
            A1=A1,E=E,lbar = lbar,A=A)

        distance = maximum(abs.(policy1-policy))

        if distance == NaN || distance == Inf
            error("Agent Problem did not converge")
        end

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

        itpc = LinearInterpolation((A,E),policy, extrapolation_bc=Line())
        iteration +=1
        if iteration == 500 || iteration > 1200
            update_policy = rand()
        elseif iteration >10000
            break
        end

    end
    return  policy
end


function innerforloop!(policy1::AbstractArray, policy_c::Function,policy_n::Function,w::Real,R::Real;
    A1=A1::Real,E=E::AbstractArray,lbar = lbar::Real,A=A::Real)
    A0::AbstractArray = copy(A1)
         for (ei,e) = enumerate(E)
            for (ai,a1) = enumerate(A1)
                policy1[ai,ei],A0[ai] = EE(a1,w,R;e=e,η=η,policy_c = policy_c)
            end

            ordem = sortperm(A0)
            itpc0 = LinearInterpolation(A0[ordem],policy1[ordem,ei],extrapolation_bc=Line())

            for ai = 1:length(A)
                if A0[1]<=A1[ai]
                    policy1[ai,ei] = itpc0(A1[ai])
                else
                    policy1[ai,ei] = η*(R*A1[ai]- A1[1] + max(w*e*lbar,0.0))
                end
            end
        end
    return policy1
end


function EE(a1::Real,w::Real,R::Real;e=E[e]::Real,
    E=E::AbstractArray, η=η::Real,policy_c=policy_c::Function,pdfE=pdfE::AbstractArray)

    i::Real = findfirst(E.==e)
    RHS1::Real = 0.0
    for e1=1:length(E)
        c1::Real = policy_c(a1,E[e1])
        a2::Real = a1star(c1,a1,E[e1],w,R)
        n1::Real = nstar(c1,E[e1],w)
        l1::Real = lbar - n1
        RHS1 += β * pdfE[i,e1]*R*uc(c1,l1)
    end

    if e > 0.0
        c::Real = (RHS1/η * ((1-η)/(η*e*w))^(-(1-μ)*(1-η)))^(-1/μ)
    else
        c = (RHS1/η *lbar^(-(1-μ)*(1-η)))^(1/(η*(1-μ)-1))
    end

    a::Real = (c+a1-max(e*w*nstar(c,e,w),0))/R

    return c,a
end
