
#CRRA utility functions


function u(c,l;η::Float64 = η,μ::Float64 = μ)
    u =0.0
    if (c<=0) || (η!=1.0 && l<=0.0)

        u= 1e15*min(c,l)+log(eps())*η
        #=if μ ==1.0
            u= log(eps(0.))+ 1e200*min(c,l) - 1
        else
            u=((eps(0.))^(1-μ) )/ (1-μ)-eps(0.) + 1e200*min(c,l)
        end =#
    elseif μ == 1.0
        u = η * log(c) + log(l^(1-η))
    else
        u = ((c^η * l^(1-η))^(1-μ) )/ (1-μ)
    end
    return u
end

# derivative of u with respect to c

function uc(c,l;η = η,μ = μ)
    if μ == 1.0 && c>0.0
        return min(η/c,1e15)
    elseif c<=0
        return (-c)*1e10+1e10
    elseif η!=1 && l <= 0
        return 0.0
    else
        return (η * c^(η-1) * l^(1-η)) * (c^η * l^(1-η))^(-μ)
    end
end

# derivative of u with respect to l
function ul(c,l;η = η,μ = μ)
    if η!=1
        if μ == 1.0
            return min((1-η)/l,1e15)
        elseif c<=0
            return 0.0
        elseif l<=0
            return -l*1e10 + 1e15
        else
            return ((1-η) * c^η * l^(-η))*(c^η * l^(1-η))^(-μ)
        end
    else
        return 0.0
    end

end


function ucc(c,l;η = η,μ = μ)
    if c<=0
        return (-c)
    elseif η!=1 && l <= 0
        return 0.0
    else
        return -(c^(η - 2)*η*l^(1-η)*(η*μ-η+1))/(c^η*l^(1-η))^μ

    end
end

function ucl(c,l;η = η,μ = μ)
    if c<=0
        return (c)
    elseif η!=1 && l <= 0
        return l
    else
        return (c^(2*η - 1)*η*l^(1 - 2*η)*μ*(η - 1))/(c^η*l^(1 - η))^(μ + 1) - (c^(η - 1)*η*(η - 1))/(l^η*(c^η*l^(1 - η))^μ)
    end
end

function ull(c,l;η = η,μ = μ)
    if η!=1
        if c<=0
            return 0.0
        elseif l<=0
            return -l
        else
            return (c^η*η*(η - 1))/(l^(η + 1)*(c^η*l^(1 - η))^μ) - (c^(2*η)*μ*(η - 1)^2)/(l^(2*η)*(c^η*l^(1 - η))^(μ + 1))
        end
    else
        return 0.0
    end

end
