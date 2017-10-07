%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       main.m
%       2017/07/03
%       Yubo Zhang
%       load many dataset in
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;
load('ORL32_indices.mat');
gnd = gnd - 5;
% load(['../' 'somedata/' 'ORL32_indices.mat']);
%   paramters and class metrix setting

X=NormalizeFea(double(X));
options=[];
options.ReducedDim=50;
[eigvector,eigvalue] = PCA(X,options);
X = X*eigvector;
originX = X;
% K = 3:2:5;
% LAMBDA = [1,10,100];
% GAMMA = [0.1,1,10,100];
K = 3;
LAMBDA = [100];
GAMMA = [0.1];
MU = 0.1:0.2:0.9;
n = size(gnd,1);
m = size(X,2);
fold = 2;  %10-fold cross validation
%times = 1;
iteration = 2;
thr=10;

meanacc = zeros(length(K),length(LAMBDA),length(GAMMA), length(MU));
best_parameters = zeros(1,4);
category = unique(gnd);
class = zeros(n,length(category));
for i = 1:n
    class(i,gnd(i)) = 1;
end 
indices = indices(1,:);
n_fold = n/fold;
%gnd = gnd(indices);


for k = 1:size(K,2)
    for gam = 1:size(GAMMA,2)
        for lam = 1:size(LAMBDA,2)
            for mu = 1:size(MU,2) 
                Flabel = zeros(n,1);
                fprintf('----parameters:  k = %d, gamma = %f, lambda = %f, mu = %f---\n' , K(k),GAMMA(gam),LAMBDA(lam),MU(mu));
                Ftestlabel=zeros(n,1);
                for i = 1:fold
                    I = eye(n);
                    X = originX;
                    tmp = zeros(n,1);
                    tmp(indices(((i-1)*n_fold+1):(i*n_fold))) = 1;
                    idx_test = logical(tmp);
                    Y = class;
                    Y(idx_test,:) = 1/length(category);
           
                    for itr = 1:iteration
                        [H,W,D] = hg_construction(X,K(k));
                        [F,R] = hg_learning(H,W,LAMBDA(lam),Y);
                        [confidence,Flabel]=max(F,[],2);
%                         ii = find(confidence>=0.1);
                        F(:)=0;
                        for idx=1:length(Flabel)
                            F(idx,Flabel(idx))=1;
                        end
                        
%                         A = getA(X(ii,:), F(ii,:)*F(ii,:)', R(ii,ii), thr, MU(mu), GAMMA(gam));
                        A = getA(X(~idx_test,:),X, F(~idx_test, :)*F(~idx_test, :)', R, thr, MU(mu), GAMMA(gam));
                        L = real(A^0.5);
                        X = X*L';
                    end
                    [H,W,D] = hg_construction(X,K(k));
                    F = hg_learning(H,W,LAMBDA(lam),Y);
                    [~,Flabel]=max(F,[],2);
                    Ftestlabel(idx_test)=Flabel(idx_test);
                end
                acc=evaluation(Ftestlabel,gnd);
                fprintf('acc: %f\n',acc);
                meanacc(k,lam,gam, mu) = acc;             
            end
        end
    end
end
[max_acc,max_loc] = max(meanacc(:));
[max_k,max_lam,max_gam,max_mu] = ind2sub(size(meanacc),max_loc);
fprintf('-------The acc is %f--------------\n-----------The best parameters is :\nK = %d, lambda = %f, gamma = %f, mu = %f--------\n',max_acc,K(max_k),LAMBDA(max_lam),GAMMA(max_gam), MU(max_mu));
