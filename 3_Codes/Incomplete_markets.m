% CALL THE PROGRAM INVARIANT.M

% In order to use this program first of all call the program invariant.m
% that calculates the average income that is going to be use to
% set the borrowing constraint and the maximum value of the asset grid. In
% accordance with the exercise the borrowing constraint will be equal to
% one year's average income and the maximum value of the asset grid will be
% equal to 3 times the average income as we can see in the parameters
% below.
% Observe that the program invariant.m not necessarily need to be used if
% you define arbitrary values for the borrowing constraint and the maximum
% value of the asset grid.

% Diego Rodrigues

%% PARAMETERS OF THE MODEL

sigma = 1.50;                           % Coefficient of risk aversion           
beta  = (0.96)^(1/6);                   % Discount factor considering the period of two months, given the anual discount factor is 0.96
prob  = [0.925 0.075; 0.5 0.5];         % Transition Matrix
el  = 0.1;                              % Low endowment - unemployed 
eh  = 1.00;                             % High endowment - employed 
Rstart = (1+0.034)^(1/6);               % Initial interest rate. The anual interest rate is 3.4 %
R1 = (1+0.034)^(1/6);                   % Interest gross rate considering the period of two months. The annual basis is 3.4%
average_income_year = average_income*6; % One year's average income. We multiply by six, since the period lenght is two months
b = -average_income_year;               % Borrowing constraint equal to one year's average income
g = 0.60;                               % Relaxation Parameter

%% ASSET GRID
  
maxast = 3*average_income;                     % Maximum value of Asset grid   
minast = -average_income_year;                 % Minimum value of Asset grid
N = 20;                                        % Number of grid points
a_grid=linspace(minast,maxast,N);              % Grid points of the asset grids
ia = a_grid(1,N)-a_grid(1,N-1);                % 'ia' is the size of increments in the Asset Grid
                         

%% STEPS TO FIND R

% Solution algorithm

% Here our objective is findind R using the algorith we saw in class contained
% 1. Guess R = Rj;
% 2. Solves households's problem using dynamic programming to find gj(a,z)
% = a' (optimal policy function for assets) and lambdaj(a,z) =
% unconditional distribution of (at,zt);
% 3. Compute e = sum(lambdaj()*gj());
% 4. If e > epsilon update rj+1 < rj (if e < epsilon, update rj+1 > rj) and go
% back to step 1. If |e| < epsilon stop.


%% COMPUTE THE VALUE FUNCTION AND THE POLICY RULE

iter   = 1;
maxiter = 50;
toler   = 0.0001;
step  = 0.05;
R = Rstart;
flag = 1;
disp('ITERATING ON R');
disp('');
disp('    liter        R       A       newstep');
while  (flag ~= 0) & (iter <= maxiter);
   util1=-10000*ones(N,N);  % utility when the endowment is high - employed    
   util2=-10000*ones(N,N);  % utility when the endowment is low - unemployed
   for i=1:N
         a=(i-1)*ia + minast;
         for j=1:N
               ap = (j-1)*ia + minast;
               % Now we are solving for consumption at each point
               c = eh + R*a - ap;
               if ap >= b & c > 0;
                  util1(j,i)=(c)^(1-sigma)/(1-sigma);
               end;
         end
         for j=1:N
               ap = (j-1)*ia + minast;
               c = el*eh + R*a - ap;
               if ap >= b & c > 0;
                  util2(j,i)=(c)^(1-sigma)/(1-sigma);
               end;
         end;
   end;
   
 % Dynamic programming of the Step 2
   
   V    = zeros(N,2); % This was constructed to contain the Value Function
   gj   = zeros(N,2); % This was constructed to contain the Policy function a' = gj(a,z)
   test1    = 10;
   test2    = 10;
   [ii,kk] = size(util1);
   
   while (test1 ~= 0) | (test2 > .1);
       for i=1:kk;
           r1(:,i)=util1(:,i)+beta*(prob(1,1)*V(:,1)+ prob(1,2)*V(:,2));
           r2(:,i)=util2(:,i)+beta*(prob(2,1)*V(:,1)+ prob(2,2)*V(:,2));
       end;

       [TV1,Tgj1]=max(r1);
       [TV2,Tgj2]=max(r2);
       Tgj=[Tgj1' Tgj2'];
       TV=[TV1' TV2'];

       test1=max(any(Tgj-gj));
       test2=max(max(abs(TV - V))');
       V=TV;
       gj=Tgj;
   end;
   gj=(gj-1)*ia + minast;
   
   
%% STATIONARY ASSET DISTRIBUTION

% Create trans, a transition matrix from state at t (row) to state at t+1 
% (column) for joint asset and labour holdings.
% The eigenvector associated with the unit eigenvalue of trans' is the
% stationary distribution

   g2=sparse(kk,kk);
   g1=sparse(kk,kk);
   for i=1:kk
       g1(i,Tgj1(i))=1;
       g2(i,Tgj2(i))=1;
   end
   trans=[ prob(1,1)*g1 prob(1,2)*g1; prob(2,1)*g2 prob(2,2)*g2];
   trans=trans';
   probst = (1/(2*N))*ones(2*N,1);
   test = 1;
   while test > 10^(-8);
       probst1 = trans*probst;
       test = max(abs(probst1-probst));
       probst = probst1;
   end; 
   
% Vectorise the decision rule to be conformable with probst 
% calculate new aggregate capital stock 

   aa=gj(:);
   meanA=probst'*aa;

%  Calculate measure over (k,s) pairs
%  lambda has same dimensions as decis

   lambda=zeros(kk,2);
   lambda(:)=probst;
 
% Calculate stationary distribution of (at,zt):
   
   probk=sum(lambda');    
   probk=probk';
 
%   Update R

   if iter == 1;
      A = meanA; 
      if meanA > 0.0;
         step = -step;
      end;
   end;
   Aold = A;
   Anew = meanA;
   if sign(Aold) ~= sign(Anew)
     step = -.5*step;
   end;
   disp([ iter R meanA step ]);
   if abs(step) >= toler;
      R = R + step;
   else;
      flag = 0;
   end;
   A = Anew;
   iter = iter+1;
end;

%% PLOTTING POLICY FUNCTION OF AP AGAINST A

    a_grid=linspace(minast,maxast,N);
    fig1 = figure('units','normalized','outerposition',[0 0 0.8 1]);
    set(fig1,'Color','white','numbertitle','off','name','Policy Function - Savings')
    plot(a_grid,gj(:,1),'b-.','LineWidth',1); hold on;
    plot(a_grid,gj(:,2),'r','LineWidth',2); hold on;
    plot(a_grid,a_grid,'k:','LineWidth',1); hold off;
    legend('Low Employment State','High Employment State','45-degree Ray','FontSize',8,'Location','SouthEast','Orientation','Vertical','Interpreter','latex');
    title('Household Savings Policy Function','FontSize',10,'Interpreter','latex');
    xlabel('$a_{i,t}$','FontSize',10,'Interpreter','latex');
    ylabel('$a_{i,t+1}$','FontSize',10,'Interpreter','latex');
    axis('tight');
    print('policyfunction','-depsc')     
    
%%   CALCULATE CONSUMPTION AND EXPECTED UTILITY

grid = a_grid';  
congood = eh*(ones(N,1)) + R*grid - grid(Tgj(:,1));
conbad  = el*eh*(ones(N,1)) + R*grid - grid(Tgj(:,2));
consum  = [congood conbad ];
cons2   = [congood.^2 conbad.^2];
meancon = sum(diag(lambda'*consum));
meancon2  = sum(diag(lambda'*cons2));
varcon = ( meancon2 - meancon^2 );
UTILITY = (consum.^(1-sigma))./(1-sigma);
UCEU2 = sum(diag(lambda'*UTILITY));

%%   RESULTS 

disp('PARAMETER VALUES');
disp('');
disp('    sigma      beta      b      el'); 
disp([ sigma beta b el]);
disp(''); 
disp('EQUILIBRIUM RESULTS ');
disp('');
disp('      R         A      UCEU     meancon    varcon');
disp([ R  meanA UCEU2 meancon varcon]);


%%   EVOLUTION OF THE AGENT

disp('SIMULATING LIFE HISTORY');
             
asset = meanA;                  % Initial level of assets
asset = 0;                      % Initial level of assets
n = 10000;                      % Number of periods to simulate
s0 = 1;                         % Initial state
hist = zeros(n-1,2);
consumption = zeros(n-1,1);
investment = zeros(n-1,1);
labincome = zeros(n-1,1);
grid = a_grid';  
[chain,state] = markov(prob,n,s0);
for i = 1:n-1;
    hist(i,:) = [ asset chain(i) ];
    I1 = round((asset-minast)/ia) ;
    I2 = round((asset-minast)/ia) + 1;
    if I1 == 0;
       I1=1;
       disp('N.B.  I1 = 0');
    end;
    if I2 > N;
       I2 = N;
       disp('N.B.  I2 > nasset');
    end;
    weight = (grid(I2,1) - asset)/ia; 
    aprime = weight*(gj(I1,chain(i))) +  (1-weight)*(gj(I2,chain(i)));
    if chain(i) == 1;
       labincome(i) = eh;
       consumption(i) = eh + R*asset - aprime;
    elseif chain(i) == 2;
       labincome(i) = eh*el;
       consumption(i) = eh*el + R*asset - aprime;
    else;
      disp('something is wrong with chain');
      chain
    end;
    asset = aprime;
    investment(i) = aprime;
    wealth = R.*investment + labincome;
end;
plot((1:n-1)',investment,(1:n-1)',consumption);
title('INVESTMENT AND CONSUMPTION');
legend('INVESTMENT','CONSUMPTION','FontSize',8,'Location','SouthEast','Orientation','Vertical','Interpreter','latex');
print('invandcons','-depsc')  
cov(consumption,investment)

plot((1:n-1)',labincome,(1:n-1)',consumption);
title('LABOR INCOME AND CONSUMPTION');
legend('LABOR INCOME','CONSUMPTION','FontSize',8,'Location','SouthEast','Orientation','Vertical','Interpreter','latex');
print('labincandcons','-depsc')  
cov(labincome,consumption)

plot((1:n-1)',labincome,(1:n-1)',investment);
title('LABOR INCOME AND INVESTMENT');
legend('LABOR INCOME','INVESTMENT','FontSize',8,'Location','SouthEast','Orientation','Vertical','Interpreter','latex');
print('labincandinvest','-depsc')
cov(labincome,investment)

plot((1:n-1)',wealth);
title('WEALTH');
legend('WEALTH','FontSize',8,'Location','SouthEast','Orientation','Vertical','Interpreter','latex');
print('wealth','-depsc')

histogram(investment)
title('HISTOGRAM OF ASSET HOLDINGS')
xlabel('Value of the Asset Holding','FontSize',10,'Interpreter','latex');
ylabel('Number of periods','FontSize',10,'Interpreter','latex')
print('histogram','-depsc')

%% INCOME DISTRIBUTION    

income =  [ (R*grid + eh)  (R*grid + eh*el) ]  ; 
[ pinc,index ] = sort(income(:));
plambda = lambda(:);
plot(pinc,plambda(index));
title('INCOME DISTRIBUTION');
xlabel('INCOME LEVEL');
ylabel('% OF AGENTS');
print('incomedistribution','-depsc')

%%  STANDARD MEASURES FOR WEALTH DISPERSION

mean_w=mean(wealth(n-1000:n-1));
stdr_w=std(wealth(n-1000:n-1));

histogram(wealth);
annotation('textbox',[.2 .3 .4 .5],...
    'String',{['Mean = ' num2str(mean_w)],['Stddev =' num2str(stdr_w)]},...
'FitBoxtoText','on');
title('Wealth, \sigma=1.5')
print('wealthdispersion','-depsc')


diary off