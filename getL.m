%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       getA.m
%       2017/03/15
%       Yubo Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function L = getL(X, Y, R, thr, lambda, gamma)
%% input:
%       X : the input data
%       Y : Y_ik=1 if S_i and S_k are in the same class, Y_ik=0 otherwise
%       R : hypergraph constraint
%       thr : the number of neighbors
%       lambda : the trade-off 
%       gamma : the parameter 
%       v : the parameter used to calculate sigma
%% output:
%       L: the update
fprintf('update L\n');
[d,l] = size(X);
e = ones(l,1);

% get neighbor matric N
ED = dist(X); % the euclidean distance
N = zeros(l);
[~,IDX] = sort(ED,2);
for i =1:l
    N(i,IDX(i,2:thr))=1;
end
% get beta
%beta = repmat(1/l,l,1);
beta = getBeta(X',N);
beta = beta./max(beta);

% get H/eta
H = N.*Y; %eta_ij=1 if S_i and S_j are neighbors in the same class, otherwise eta=0


% %% GET M
% C1 = diag((e*e'-Y)*e)*H;
% C0 = gamma*diag(H*e)*(e*e'-Y);
% C = C1-C0;
% Dc = diag(C*e);
% M = X*(Dc-C)*X';%??
%% GET B
W = (diag(beta)*R).*N;
Dw = diag(W*e);
B = X*(Dw-W)*X';%??
%% calculate L to minimize (1-lambda)*tr(MA)+lambda*tr(B'A) by iteration
% f'(L) = (1-lambda)*(LM+LM')+lambda*(LB'+LB)
L = pca(X)';
options.MaxFunEvals=400;
[L,loss,~,det] = minFunc(@(L) lossGradient(L(:),X,B,N,Y,lambda, gamma),L(:),options);
L = reshape(L,[d,d]);


% alpha = 1e-11; %learning rate
% T = 500; %iteration time
% dfA = (1-lambda)*M'+lambda*B';
% 
% for t=1:T
%     %A = A-alpha*dfA;
%     %利用几何结构把梯度拉回到李代数里面
%     Gt = A^-.5*dfA*A^-.5;
%     A = (A^.5)*exp(-alpha*Gt)*(A^.5);
%     A = (A+A')/2;
%     [U,V] = eig(A);
%     V(V<0) = 0;
%     V = V+diag(ones(d,1)*1e-50);
%     A = U*V*U';
%     fA = (1-lambda)*trace(M*A)+lambda*trace(B*A);
%     fprintf('iteration times:%d f(A)=%f\n',t,fA);
% end

