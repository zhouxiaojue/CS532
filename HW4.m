addpath /Users/kaeda/Documents/uw/2015Fall/CS532/HW4
load jesterdata.mat
load newuser.mat

%problem1
index=logical(b~=-99);
X_small=X(index,1:20);
beta=inv(X_small'*X_small)*X_small'*b(index);
prediction=X(:,1:20)*beta;

compare=[prediction trueb];
plot(trueb-prediction)

%no, its close that the prediction is 
%3.5155    6.2600   

%problem2
X_train=X(index,:);
beta=(X_train'/(X_train*X_train'))*b(index);
prediction=X*beta;
compare=[prediction trueb];
plot(trueb-prediction)

%problem3
%find the largest weights of those two users
%for one user
[maxbeta,ind]=max(abs(beta));
%it's the 2503th user

prediction_one=X(:,2503)*beta(2503);
MSE_one=mean(trueb-prediction_one);
plot(trueb-prediction_one)
%the error is the way off than previous
%-1.7032

%for two users
[sortbeta,indexsort]=sort(abs(beta));
prediction_two=X(:,[indexsort(end) indexsort(end-1)])*beta([indexsort(end) indexsort(end-1)]);
MSE_two=mean(trueb-prediction_two);
%the error is  -1.7038. even worse than last one. Since maybe the weights
%are pretty close to each other.


%problem4

[U,S,V] = svd(X,'econ');
rank(S);
plot(svd(X))
%seems like only 10-20 dimensions are important
%size(S)=100x100

%problem5
scatter3(V(:,1)',V(:,2)',V(:,3)');
%typical jokes
%it's mostly along a plane and like a spindle. tells are maybe there are
%two kinds of jokes that are representative of the rest.
scatter3(U(:,1)',U(:,2)',U(:,3)');
%typical users
%it's mostly along a one-dimensional line. Tells us most user's taste are
%the same.

%problem6

%looking for v 
tol=1e-3;
X_i=X'*X;
b=X_i(:,1);
b1=ones(length(X),1);
i=0;
while norm(b-b1,1)>tol
    b=b1;
    b1=X_i*b/(norm(X_i*b));   
    i=i+1;
    if i>10000
        break
    end
end

%b1 is what we are looking for 
%it works because it finding vector that will not change. since any other
%eigenvector divided by dominant eigenvector will be smaller than 1.
%it's the absolute value of the eigen value calculated by maltab's SVD

%looking for u
X_j=X*X';
b=X_j(:,1);
b1=ones(length(X_j),1);
i=0;
while norm(b-b1,1)>tol
    b=b1;
    b1=X_j*b/(norm(X_j*b));   
    i=i+1;
    if i>10000
        break
    end
end 
%problem7
%if multiplying by all 0. it fails