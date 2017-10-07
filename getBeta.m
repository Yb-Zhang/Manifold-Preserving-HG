function [ beta ] = getBeta( X, N )
%getBeta
%   X; the input data
%   N: the neighborhood list
n = size(X,1);
d = size(X,2);
beta = zeros(n,1);
for i=1:n
    idx = N(i,:); % the index of neighbors of point i
    neighbors = diag(idx)*X;
    neighbors (all(neighbors == 0, 2),:) = []; %ȫ������Ϊ�գ�����ȥ�� 
    pdf = akde(neighbors,X(i,:)); % the density of point i
    beta(i,:)=pdf;
end
end

