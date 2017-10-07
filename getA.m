%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       getA.m
%       2017/03/15
%       Yubo Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function A = getA(XL, X, Y, R, thr, lambda, gamma)
%% input:
%       XL:the input labeled data
%       X : the input data
%       Y : Y_ik=1 if S_i and S_k are in the same class, Y_ik=0 otherwise
%       R : hypergraph constraint
%       thr : the number of neighbors
%       lambda : the trade-off 
%       gamma : the parameter 
%       v : the parameter used to calculate sigma
%% output:
%       L: the update
%fprintf('update L\n');
[l,d] = size(X);
e = ones(l,1);

% get neighbor matric N
ED = dist(X'); % the euclidean distance
N = zeros(l);
[~,IDX] = sort(ED,2);
for i =1:l
    N(i,IDX(i,2:thr))=1;
end
[iL, ~] = size(XL);
eL = ones(iL, 1);
nL = zeros(iL);
EDL = dist(XL');
[~,idxL] = sort(EDL,2);
for i = 1:iL
        nL(i,idxL(i,2:thr)) = 1;
end
nL = nL + nL';
nL = double(logical(nL));
% get beta
beta = getBeta(X,N);
beta = beta./max(beta);

% get H/eta
H = nL.*Y; %eta_ij=1 if S_i and S_j are neighbors in the same class, otherwise eta=0


%% GET M
C1 = diag((eL*eL'-Y)*eL)*H;
C0 = gamma*diag(H*eL)*(eL*eL'-Y);
C = C1-C0;
Dc = diag(C*eL);
M = XL'*(Dc-C)*XL;
%% GET B
W = (diag(beta)*R).*N;
Dw = diag(W*e);
B = X'*(Dw-W)*X;
%% calculate L to minimize (1-lambda)*tr(MA)+lambda*tr(B'A) by iteration

alpha = 1e-5; %learning rate
T = 400; %iteration time
dfA = (1-lambda)*M'+lambda*B';
dfA = (dfA+dfA')/2;
dfA = pinv(dfA);
dfA = (dfA+dfA')/2;
A = eye(d);
for t=1:T
    %利用几何结构把梯度拉回到李代数里面
    prj=real(A^.5);
    invprj=inv(prj);
    Gt = invprj*dfA*invprj;
    Gt = (Gt+Gt')/2;
    [V,dg]=eig(Gt);
    edg=sum(dg);

    edg((edg<0)&abs(edg)<1e-2)=-1e-2;    % 阈值
     
    A = prj*V*diag(exp(-alpha./edg))*V'*prj;
    A = real((A+A')/2);
    fA = (1-lambda)*trace(M*A)+lambda*trace(B*A);
    %fprintf('iteration times:%d f(A)=%f\n',t,fA);
end

