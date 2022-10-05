
using Distributions
function Tauchen(ρ,σ,N,μ,m)
    #This function is used to discretize an AR(1) process following Tauchen method
    # y_(t+1) = μ(1-ρ) + ρy_(t) + ϵ
    # ϵ ∼ N (0, σ^2)
    # N represents the number of y states

    if N>1
        yN= μ + m*(σ^2/(1-ρ^2))^(1/2) # This representes the maximum value of y
        y1= μ - m*(σ^2/(1-ρ^2))^(1/2) # This representes the minimum value of y
        Δ = (yN -y1)/(N-1) #This is the distance between each y
        y = y1:Δ:yN # This is the vector of possible states

    d = Normal()

    pdfY=ones(N,N) #This preallocate memory and create the transition matrix in the followimg loop
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


## CALIBRATION FOR THE STOCHASTIC PROCESS
ρ = 0.9;
μ = 1;
σ = 0.5;
N = 3;
m =3;

Π, y = Tauchen(ρ,σ,N,μ,m)
