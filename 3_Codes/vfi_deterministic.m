% Macro III
% Canonical Growth Model with No Labour and No Investment Constraint
% DETERMINISTIC MODEL
% Equations:
% V(k) = max_{c,k} ln(c) + beta*V(k')
% where
% c + k' = k^(alpha) + (1-delta)*k
% c>=0 and k'>=0
% SOLUTION METHOD: VALUE FUNCTION ITERATION (Discretisation) and Howards
% Improvement algorithm
% Diego de Sousa Rodrigues
% Fall, 2021

clear; clc;

%% Define Preliminaries

% Parameters
alpha = 0.36;   % Capital Share in Production Function
beta  = 0.99;   % Discount Factor
delta = 0.025;  % Depreciation Rate

% Utility Function 
u = @(ct) log(ct);

% Production Function 
F = @(kt) kt^(alpha);

% Define the Steady State
k_ss = (alpha/((1/beta) - (1-delta)))^(1/(1-alpha));  % Steady State Capital Stock
y_ss = F(k_ss);                     % Steady State Output
c_ss = y_ss - delta*k_ss;           % Steady State Consumption

%% Define the Grid for the Endogenous State Variable: Capital
nk = 501;                            % Number of Grid Points
kmin = 0.2*k_ss; kmax = 1.8*k_ss;     % Bounds for Grid
kg = [kmin:(kmax-kmin)/(nk-1):kmax];  % Equally Spaced Grid for Capital

%% Build the 2-Dimensional Contemporaneous Utility Grid
% Within it, ensure that capital and consumption never go negative

Ut = zeros(nk,nk);

for ii = 1:nk       % Loop Over Capital Today
    for jj = 1:nk   % Loop Over Capital Tomorrow
        k  = kg(ii);
        kp = kg(jj);
        % Solve for Consumption at Each Point
        c = F(k) + (1-delta)*k - kp;
        if (kp < 0)||(c < 0)   
            % If Tomorrow's Capital Stock or Today's Consumption is Negative
            Ut(ii,jj) = -9999999999; 
            % Numerical Trick to ensure that the Value Function is never 
            %    optimised at these points
        else
            Ut(ii,jj) = u(c);
        end
    end
end

%% Define an Initial Guess for the Value Function
V0 = ones(nk,1);    % nk x 1 vector of initial guess
V1 = zeros(nk,1);   % nk x 1 vector for optimal value function for given position on capital grid
Objgrid = zeros(nk,nk);  % nk x nk matrix where rows denote today's capital and columns tomorrow's

% Value Function Iteration
tol  = 0.0001;
err  = 2;
iter = 0;

while err > tol
    for ii = 1:nk       % Loop over today's capital stock
        for jj = 1:nk   % Loop over tomorrow's capital stock
            Objgrid(ii,jj) = Ut(ii,jj) + beta*V0(jj);
        end
        [V1(ii,1),PF(ii,1)] = max(Objgrid(ii,:));
    end
    for hh=1:20 % Howards' improvement algorithm
        for ii=1:nk,
            if hh==1
                V11(ii,1)=Ut(ii,PF(ii,1))+beta*V1(PF(ii,1),1);
            elseif hh>1
                V11(ii,1)=Ut(ii,PF(ii,1))+beta*V11(PF(ii,1),1);
            end
        end
    end
    V1=V11;
    iter = iter + 1;
    err  = norm(V1(:) - V0(:));
    iter10 = mod(iter,10);
    if iter10 == 0
        display(['Number of Iterations ',num2str(iter)]);
    end
    V0   = V1;
end

%% Build Policy Function for Consumption
CF = zeros(size(PF));

for j = 1:nk
	k  = kg(j);         % Capital Today
    kp = kg(PF(j,1));   % Capital Tomorrow
    
    % Solve for Consumption
    CF(j,1) = F(k) + (1-delta)*k - kp;
end

%% Plot the Policy Function
fig1 = figure('units','normalized','outerposition',[0 0 0.8 1])
    set(fig1,'Color','white','numbertitle','off','name','Policy Function - Capital')
    plot(kg,kg(PF(:,1)),'k','LineWidth',2)
    hold on
    plot(kg,kg,'k:','LineWidth',1)
    hold off
    xlabel('k Today')
    ylabel('k Tomorrow')
    title('Capital Policy Function')    