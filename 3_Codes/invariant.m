prob = [0.925 0.075; 0.5 0.5];
% Calculate the invariant distribution of Markov chain by simulating the
% chain to reach a long-run level
Trans= prob';
probst = (1/2)*ones(2,1); % initial distribution of states
test = 1;

while test > 10^(-8);
	probst1 = Trans*probst;
	test=max(abs(probst1-probst));
	probst = probst1;   
end

v = probst1';

income = [1 , 0.1];
average_income = v*income';