%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       hg_construction.m
%       2017/03/17
%       Yubo Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [H,W,D] = hg_construction(X,k)
n = size(X,1);
D = EuDist2(X);
H = zeros(n);
% D
% Tmp = X*X';
% Tmp_i = D+diag(Tmp)';
% Tmp_j = D+diag(Tmp);
% D = Tmp_i+Tmp_j-2*Tmp;


[B,I] = sort(D);
for i = 1:n
    H(I(1:k+1,i),i) = 1;
end
W = eye(n)*1/n;
