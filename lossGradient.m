function [ loss, df ] = lossGradient(L,X,B,N,Y,lambda,gamma  )
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
[d,n] = size(X);
L = reshape(L,[numel(L)/d d]);
loss = 0.0;
md = [];
ed = cell(n);
df = zeros(d);
for i=1:n
    for j=find(N(i,:).*Y(i,:)~=0)
        md(i,j) = (X(:,i)-X(:,j))'*(L'*L)*(X(:,i)-X(:,j));
        ed{i,j} = (X(:,i)-X(:,j))*(X(:,i)-X(:,j))';
    end
end
for i=1:n
    for j=find(N(i,:).*~Y(i,:)~=0)
        md(i,j) = (X(:,i)-X(:,j))'*(L'*L)*(X(:,i)-X(:,j));  
        ed{i,j} = (X(:,i)-X(:,j))*(X(:,i)-X(:,j))';
    end
end
for i=1:n
    for j=find(N(i,:).*Y(i,:)~=0)
        for k=find(N(i,:).*~Y(i,:)~=0)
            if 1+md(i,j)-gamma*md(i,k)>0
                loss = loss+1+md(i,j)-gamma*md(i,k);
                df = df+2*L*ed{i,j}-gamma*2*L*ed{i,k};
            end            
        end
    end
end
loss = (1-lambda)*loss+lambda*(vec(B)'*vec(L'*L));
df = (1-lambda)*df+lambda*(L*B'+L*B);
df = vec(df);
end

