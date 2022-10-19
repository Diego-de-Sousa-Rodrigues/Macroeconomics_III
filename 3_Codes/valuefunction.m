%%% VALUE FUNCTION ITERATION STEPS

% DETERMINISTIC MODEL

% Equations:
% V(k) = max_{c,k} ln(c) + beta*V(k')
% where
% c + k' = theta* k^(alpha) + (1-delta)*k
% c>=0 and k'>=0
% SOLUTION METHOD: VALUE FUNCTION ITERATION 

clear; clc;

%% Define Preliminaries

% Parameters
alpha = 0.40;   % Capital Share in Production Function
beta  = 0.60;   % Discount Factor
delta = 1;      % Depreciation Rate
theta = 10;    % Productivity factor  

% Utility Function 
u = @(ct) log(ct);

% Production Function 
F = @(kt) theta * kt^(alpha);

% Define the Steady State
k_ss =  ((theta* alpha)/((1/beta) - (1-delta)))^(1/(1-alpha));          % Steady State Capital Stock
y_ss = F(k_ss);                                                         % Steady State Output
c_ss = y_ss - delta*k_ss;                                               % Steady State Consumption

%% Define the Grid for the Endogenous State Variable: Capital

nk = 500;                               % Number of Grid Points
kmin = 2; kmax = 10;                  % Bounds for Grid
kg = kmin:(kmax-kmin)/(nk-1):kmax;    % Equally Spaced Grid for Capital

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
            % optimised at these points
        else
            Ut(ii,jj) = u(c);
        end
    end
end

%% Define an Initial Guess for the Value Function
V0 = ones(nk,1);    % nk x 1 vector of initial guess
V1 = zeros(nk,1);   % nk x 1 vector for optimal value function for given position on capital grid
V2 = zeros (nk,1);  % nk X 1 vector for optimal value function for given position on capital grid
V3 = zeros (nk,1);  % nk X 1 vector for optimal value function for given position on capital grid
V4 = zeros (nk,1);  % nk X 1 vector for optimal value function for given position on capital grid
V5 = zeros (nk,1);  % nk X 1 vector for optimal value function for given position on capital grid
V6 = zeros (nk,1);  % nk X 1 vector for optimal value function for given position on capital grid
V7 = zeros (nk,1);  % nk X 1 vector for optimal value function for given position on capital grid
V8 = zeros (nk,1);  % nk X 1 vector for optimal value function for given position on capital grid
V9 = zeros (nk,1);  % nk X 1 vector for optimal value function for given position on capital grid
V10 = zeros (nk,1); % nk X 1 vector for optimal value function for given position on capital grid
Objgrid = zeros(nk,nk);  % nk x nk matrix where rows denote today's capital and columns tomorrow's

% Value Function Iteration
tol  = 0.00001;
err  = 2;
iter = 0;

 for ii = 1:nk          % Loop over today's capital stock
        for jj = 1:nk   % Loop over tomorrow's capital stock
            Objgrid(ii,jj) = Ut(ii,jj) + beta*V0(jj);
        end
        [V1(ii,1),PF(ii,1)] = max(Objgrid(ii,:));
 end
 
 for ii = 1:nk          % Loop over today's capital stock
        for jj = 1:nk   % Loop over tomorrow's capital stock
            Objgrid(ii,jj) = Ut(ii,jj) + beta*V1(jj);
        end
        [V2(ii,1),PF(ii,1)] = max(Objgrid(ii,:));
 end
 
for ii = 1:nk          % Loop over today's capital stock
        for jj = 1:nk  % Loop over tomorrow's capital stock
            Objgrid(ii,jj) = Ut(ii,jj) + beta*V2(jj);
        end
        [V3(ii,1),PF(ii,1)] = max(Objgrid(ii,:));
end 
 
for ii = 1:nk          % Loop over today's capital stock
        for jj = 1:nk  % Loop over tomorrow's capital stock
            Objgrid(ii,jj) = Ut(ii,jj) + beta*V3(jj);
        end
        [V4(ii,1),PF(ii,1)] = max(Objgrid(ii,:));
end  
 
for ii = 1:nk          % Loop over today's capital stock
        for jj = 1:nk  % Loop over tomorrow's capital stock
            Objgrid(ii,jj) = Ut(ii,jj) + beta*V4(jj);
        end
        [V5(ii,1),PF(ii,1)] = max(Objgrid(ii,:));
end  
 
for ii = 1:nk          % Loop over today's capital stock
        for jj = 1:nk  % Loop over tomorrow's capital stock
            Objgrid(ii,jj) = Ut(ii,jj) + beta*V5(jj);
        end
        [V6(ii,1),PF(ii,1)] = max(Objgrid(ii,:));
end 

 for ii = 1:nk          % Loop over today's capital stock
        for jj = 1:nk   % Loop over tomorrow's capital stock
            Objgrid(ii,jj) = Ut(ii,jj) + beta*V6(jj);
        end
        [V7(ii,1),PF(ii,1)] = max(Objgrid(ii,:));
 end 
 
 for ii = 1:nk          % Loop over today's capital stock
        for jj = 1:nk   % Loop over tomorrow's capital stock
            Objgrid(ii,jj) = Ut(ii,jj) + beta*V7(jj);
        end
        [V8(ii,1),PF(ii,1)] = max(Objgrid(ii,:));
 end 
 
 for ii = 1:nk          % Loop over today's capital stock
        for jj = 1:nk   % Loop over tomorrow's capital stock
            Objgrid(ii,jj) = Ut(ii,jj) + beta*V8(jj);
        end
        [V9(ii,1),PF(ii,1)] = max(Objgrid(ii,:));
 end 

  for ii = 1:nk          % Loop over today's capital stock
        for jj = 1:nk    % Loop over tomorrow's capital stock
            Objgrid(ii,jj) = Ut(ii,jj) + beta*V9(jj);
        end
        [V10(ii,1),PF(ii,1)] = max(Objgrid(ii,:));
  end 
 
%% Build Policy Function for Consumption
CF = zeros(size(PF));

for j = 1:nk
	k  = kg(j);         % Capital Today
    kp = kg(PF(j,1));   % Capital Tomorrow
    
    % Solve for Consumption
    CF(j,1) = F(k) + (1-delta)*k - kp;
end


%% Plot the Policy Function for Capital
fig1 = figure('units','normalized','outerposition',[0 0 0.8 4])
    set(fig1,'Color','white','numbertitle','off','name','Policy Function - Capital')
    plot(kg,kg(PF(:,1)),'k','LineWidth',3, 'Color', [0.7 0 0])
    hold on
    plot(kg,kg,'k:','LineWidth',1)
    hold off
    xlabel('K_{t}','FontSize',16)
    ylabel('K_{t+1}','FontSize',16)
    title('Capital Policy Function')
    legend('Policy Function','45-degree Ray','FontSize','Orientation','Vertical','Location','SouthEast')
    
%% Plot the Policy Function for Consumption
fig2 = figure('units','normalized','outerposition',[0 0 0.8 4])
    set(fig2,'Color','white','numbertitle','off','name','Policy Function - Consumption')
    plot(kg,CF,'C','LineWidth',3, 'Color', [0 0 0.7])
    xlabel('K_{t}','FontSize',16)
    ylabel('Consumption','FontSize',16)
    title('Consumption Policy Function')
    legend('Policy Function','FontSize','Orientation','Vertical','Location','SouthEast') 
  
%% Plot the Value Function
fig3 = figure('units','normalized','outerposition',[0 0 0.8 4])
    set(fig3,'Color','white','numbertitle','off','name','Policy Function - Consumption')
    plot(kg,V1,'C','LineWidth',3, 'Color', [0 0.7 0])
    hold on
    plot(kg,V2,'C','LineWidth',3, 'Color', [0 0.7 0])
    hold on
    plot(kg,V3,'C','LineWidth',3, 'Color', [0 0.7 0])
    hold on
    plot(kg,V4,'C','LineWidth',3, 'Color', [0 0.7 0])
    hold on
    plot(kg,V5,'C','LineWidth',3, 'Color', [0 0.7 0])
    hold on
    plot(kg,V6,'C','LineWidth',3, 'Color', [0 0.7 0])
    hold on
    plot(kg,V7,'C','LineWidth',3, 'Color', [0 0.7 0])
    hold on
    plot(kg,V8,'C','LineWidth',3, 'Color', [0 0.7 0])
    hold on
    plot(kg,V9,'C','LineWidth',3, 'Color', [0 0.7 0])
    hold on
    plot(kg,V10,'C','LineWidth',3, 'Color', [0 0.7 0])
    hold off
    xlabel('K_{t}','FontSize',16)
    ylabel('Value Function','FontSize',16)
    title('Value Function')
    legend('Value Function','Orientation','Vertical','Location','SouthEast')

        
    
