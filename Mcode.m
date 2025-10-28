%-------------------------------------------------------------------------%
% Estimation of the generalized accelerated hazards models based on case-cohort interval-censored outcomes in the presence of informative censoring
% Author: Lou YC
%-------------------------------------------------------------------------%
clear; clc; close all; opt = optimset('Display','off');
opt3 = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton', 'OptimalityTolerance', 1e-20, 'StepTolerance', 1e-20, 'MaxFunctionEvaluations',1e6, 'MaxIterations', 1e6);
n = 2000; ps = 0.2; pc1 = 1; m = 4; B = 100; nrep = 1000;
bex = 0.5; be01 = [0.5;-0.5]; be02 = -0.2; alp0 = [0.2;0.2]; be0 = [bex;be01;be02]; pdim = size(be0,1);
ustar_a = 1; ustar_b = 2; summary = zeros(nrep,2*pdim); sumflag = zeros(nrep,1);
for rep = 1:nrep    
x = random('Binomial',1,0.5,n,1); z = [random('Normal',0,1,n,1),random('Uniform',-1,1,n,1)];
ustar = random('Gamma',ustar_a,ustar_b,n,1); u = log(ustar);  tau = random('Uniform',3,4,n,1); Lamb_ih_tau = tau .* exp(z*alp0 + u);
K = zeros(n,1); obsrvu = zeros(n,30); obsrvu_set = nan(n,30);
for ii = 1:n
while K(ii)==0, K(ii) = random('Poisson', Lamb_ih_tau(ii)); end
obsrvu(ii,1:K(ii)) = random('Uniform',0,tau(ii),1,K(ii)); obsrvu(ii,1:K(ii)) = sort(obsrvu(ii,1:K(ii)),2); obsrvu_set(ii,1:K(ii)) = obsrvu(ii,1:K(ii));
if K(ii)>0, obsrvu(ii,(K(ii)+1):end) = obsrvu(ii,K(ii)); end
end
obsrvu_add0 = [zeros(n,1),obsrvu];
tt = random('Uniform',0,1,n,1); T = 35 * (exp(-log(1-tt)./exp(z*be01+u*be02))-1)./exp(x*bex);
Ldel = (T>obsrvu_add0); Rdel = (T<=obsrvu_add0); del = Ldel(:,1:end-1)&Rdel(:,2:end);
indR = ~sum(del,2); indL = del(:,1); Lint = zeros(n,1); Rint = zeros(n,1); 
for ii = 1:n
if indR(ii)
    Lint(ii) = max(obsrvu_set(ii,:)); Rint(ii) = Lint(ii) + 0.1;
else
    idx = find(del(ii,:)==1); Lint(ii) = obsrvu_add0(ii,idx); Rint(ii) = obsrvu_add0(ii,idx+1);
end
end
subc_id = random('Binomial',1,ps,[n,1]); cas1_id = ~indR;
cas1_in_subc_id = all([subc_id, cas1_id], 2); cas1_nin_subc_id = all([cas1_id, not(subc_id)], 2); ncas1_in_subc_id = all([subc_id, not(cas1_id)], 2);
sup1_id = all([cas1_nin_subc_id, random('Binomial',1,pc1,[n,1])], 2); cc_dis1_id = any([subc_id, sup1_id], 2);
w_dis1 = cas1_in_subc_id * 1 + ncas1_in_subc_id / ps + sup1_id / pc1; z_nan = z; z_nan(not(cc_dis1_id)) = NaN;
u_all = reshape(obsrvu_set,[],1); u_set = unique(u_all); u_set = u_set(~isnan(u_set)); d = sum((u_set == u_all'),2);  R = zeros(size(d));
for ii = 1:size(u_set,1), R(ii) = sum(sum((obsrvu_set <= u_set(ii)).*(u_set(ii) <= tau),2)); end
Lamb0h = [0;cumprod(1-d(2:end)./R(2:end),'reverse')]; z_tidle = [ones(n,1),z_nan]; Lamb0h_tau = zeros(n,1);
for ii = 1:n
if tau(ii) >= max(u_set)
    Lamb0h_tau(ii) = 1; 
else
    Lamb0h_tau(ii) = max(Lamb0h(tau(ii)<=u_set));
end
end
expualphat = fsolve(@(para) sum((w_dis1.*z_tidle.*(K./Lamb0h_tau-exp(para(1))*exp(z_nan*para(2:end))))',2,'omitnan'), [log(ustar_a*ustar_b);alp0], opt);
alphat = expualphat(2:end); uhat = log(K./(Lamb0h_tau.*exp(z_nan*alphat)));
LRmax = max([Lint.*exp(x*bex);Rint.*exp(x*bex)])+0.1; bl1 = zeros(n,(m+1));   br1 = bl1;
for ii = 0:m
   bl1(:,(ii+1)) = bern(ii,m,0,LRmax,Lint.*exp(x*bex)); br1(:,(ii+1)) = bern(ii,m,0,LRmax,Rint.*exp(x*bex));
end
Lambl_true = log(1+Lint.*exp(x*bex)/35); Lambr_true = log(1+Rint.*exp(x*bex)/35);
phl01 = fminunc(@(x)sum((Lambl_true-bl1*cumsum(exp(x))).^2),zeros((m+1),1), opt); phr01 = fminunc(@(x)sum((Lambr_true-br1*cumsum(exp(x))).^2),zeros((m+1),1), opt); 
ph0 = (phl01+phr01)/2; w = [z_nan,uhat];
[ml,fval,exitflag] = fminunc(@(para) lik_GAH(para,m,pdim-1,LRmax,x,w,Lint,Rint,~indL,~indR,w_dis1), [bex;be01;be02;ph0], opt3);
sumflag(rep) = exitflag; beh = ml(1:pdim); summary_B = zeros(size(ml,1),B);
for boot = 1:B
    u_w = random('Uniform',0.5,1.5,n,1); new_w_dis1 = w_dis1 .* u_w; 
    expualphatB = fsolve(@(para) sum((new_w_dis1.*z_tidle.*(K./Lamb0h_tau-exp(para(1))*exp(z_nan*para(2:end))))',2,'omitnan'), [log(ustar_a*ustar_b);alp0], opt);
    alphatB = expualphatB(2:end); uhatB = log(K./(Lamb0h_tau.*exp(z_nan*alphatB))); wB = [z_nan,uhatB];
    [ml1B,fval,exitflag] = fminunc(@(para) lik_GAH(para,m,pdim-1,LRmax,x,w,Lint,Rint,~indL,~indR,new_w_dis1), [bex;be01;be02;ph0], opt3);
    summary_B(:,boot) = ml1B;
end
dih = std(summary_B,0,2)*sqrt(12); seb = dih(1:pdim); res = [beh',seb'];
if not(isreal(res)), res = []; end
summary(rep,:) = res;
end
rs = size(summary,1); biasb = mean(summary(:,1:pdim))'-be0; sseb = std(summary(:,1:pdim))'; seeb = mean(summary(:,(pdim+1):(2*pdim)))';
cpb = mean((repmat(be0',rs,1)>=summary(:,1:pdim)-1.96*summary(:,(pdim+1):(2*pdim)))&(repmat(be0',rs,1)<=summary(:,1:pdim)+1.96*summary(:,(pdim+1):(2*pdim))))';
tabres = [biasb,sseb,seeb,cpb]; T = table(biasb, sseb, seeb, cpb); T.Properties.VariableNames = {'Bias' 'SSE' 'SEE' 'CP'}; 
T.Properties.RowNames = {'betax', 'beta11', 'beta12' , 'beta2'};
disp("Results:");
disp(T);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function b = bern(j,p,l,u,t)
b = mycombnk(p,j)*(((t-l)/(u-l)).^j).*((1-(t-l)/(u-l)).^(p-j));
end

function m = mycombnk(n,k)
if nargin < 2, error('Too few input parameters'); end
s = isscalar(k) & isscalar(n);
if (~s), error('Non-scalar input'); end
ck = k > n;
if (ck), error('Invalid input'); end
z = k >= 0 & n > 0;
if (~z), error('Negative or zero input'); end
m = factorial(n)/(factorial(k)*factorial(n-k));
end

function output = lik_GAH(para,m,p,LRmax,x1,w1,L1int,R1int,indL1,indR1,w_dis1)
bex = para(1); bet = para(2:p+1); 
phi1 = para((p+2):(p+2+m)); ep1 = cumsum(exp(phi1));
LRmax = max([L1int.*exp(x1*bex);R1int.*exp(x1*bex)])+0.1;
bl1 = zeros(size(x1,1),(m+1));   br1 = bl1;
for ii = 0:m
   bl1(:,(ii+1)) = bern(ii,m,0,LRmax,L1int.*exp(x1*bex));
   br1(:,(ii+1)) = bern(ii,m,0,LRmax,R1int.*exp(x1*bex));
end
gLl1 = exp(-indL1.*exp(w1*bet).*(bl1*ep1)); gLr1 = exp(-exp(w1*bet).*(br1*ep1));
l1 = gLl1-indR1.*gLr1; output = - sum(w_dis1 .* log(l1), 'omitnan');
end
