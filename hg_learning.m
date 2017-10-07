%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       hh_learning.m
%       2017/03/06
%       Yubo Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [F, R] = hg_learning(H,W,lambda,Y)
n = size(H,1);
D_v = sum(H*W,2);
D_e = sum(H);
I = eye(n);
Zeta = diag(D_v.^(-1/2))*H*W*diag(D_e.^-1)*H'*diag(D_v.^(-1/2));
Delta = I-Zeta;
R = inv(I+1/lambda*Delta);
F = R*Y;