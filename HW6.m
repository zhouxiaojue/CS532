addpath /Users/kaeda/Documents/uw/2015Fall/CS532/HW6
load BreastCancer.mat

%Question 2
%a)
diff=10;
lambda=5;
beta=ones(size(X,2),1);
tao=1e-5;
run=0;
while diff>1e-2
   z=beta-tao*X'*(X*beta-y);
   beta_1=z-((lambda*tao)/2)*sign(z);
   diff=norm(beta_1-beta,2);
   beta=beta_1;
   run=run+1;
   if run>10000
       break
   end
end

%b)
lambdas=[0.01 0.1 1 1.5 3 15 30 80];
resid=[];
beta_norm=[];
X_train=X(1:100,:);
y_train=y(1:size(X_train,1));
betaall=[];
for i=1:length(lambdas)
    lambda=lambdas(i);
    diff=10;
    beta=ones(size(X_train,2),1);
    tao=1e-5;
    run=0;
    while diff>1e-2
        z=beta-tao*X_train'*(X_train*beta-y_train);
        beta_1=z-((lambda*tao)/2)*sign(z);
        diff=norm(beta_1-beta,2);
        beta=beta_1;
        run=run+1;
        if run>10000
            break
        end
    end
    resid(i)=norm(X_train*beta_1-y_train,2);
    beta_norm(i)=norm(beta_1,1);
    betaall=[betaall beta_1];
end
axis([0 100 0 1e4])
plot(beta_norm,resid)
xlabel('beta_norm')
ylabel('resid')
% Here, when the norm starts to increase, residual starts to decrease.
%Here the optimal lambda seems to be 

%c
error=[];
beta_nonzero=[];
for i=1:length(lambdas)
    beta_1=betaall(:,i);
    error(i)=mean(logical(sign(X_train*beta_1)~=y_train));
    beta_nonzero(i)=sum(logical(beta_1>1e-6));
end
axis([0 100 0 1e4])
plot(beta_nonzero,error)
xlabel('beta_nonzero')
ylabel('error')

%d
lambdas=[0.01 0.1 1 1.5 3 15 30 80];
resid=[];
beta_norm=[];
X_test=X(101:295,:);
y_test=y(1:size(X_test,1));
beta_all=[];
for i=1:length(lambdas)
    lambda=lambdas(i);
    diff=10;
    beta=ones(size(X_test,2),1);
    tao=1e-5;
    run=0;
    while diff>1e-2
        z=beta-tao*X_test'*(X_test*beta-y_test);
        beta_1=z-((lambda*tao)/2)*sign(z);
        diff=norm(beta_1-beta,2);
        beta=beta_1;
        run=run+1;
        if run>10000
            break
        end
    end
    resid(i)=norm(X_test*beta_1-y_test,2);
    beta_norm(i)=norm(beta_1,1);
    beta_all=[beta_all beta_1];
end
axis([0 100 0 1e4])
plot(beta_norm,resid)
xlabel('beta_norm')
ylabel('resid')

error=[];
beta_nonzero=[];
for i=1:length(lambdas)
    beta_1=beta(i);
    error(i)=mean(logical(sign(X_test*beta_1)~=y_test));
    beta_nonzero(i)=sum(logical(beta_1>1e-6));
end
axis([0 100 0 1e4])
plot(beta_nonzero,error)
xlabel('beta_nonzero')
ylabel('error')

%e

index=randperm(length(y));
newX=X(index,:);
newy=y(index,:);
sets=5;
n=29; %Cross Validation sets with number of elements=29
one_block=[ones(1,n),zeros(1,length(y)-n)];
CV_blocks=[];
for i=1:sets
    CV_blocks=[CV_blocks circshift(one_block',(i-1)*n,1)];
end

n=30; %=30
one_block=[ones(1,n),zeros(1,length(y)-n)];
for i=6:10
    CV_blocks=[CV_blocks circshift(one_block',(i-1)*n,1)];
end
%use eight sets to set parameter and rest one set for testing parameter. 
%test=test(test~=2)
sets=10;
indexcvall=1:10;
parameters=[0.01 0.1 0.5 1 1.5 3 15 30];
erroralllasso=zeros(1,sets);
errorallridge=zeros(1,sets);
errorpcvlasso=zeros(1,sets-1);
errorpcvridge=zeros(1,sets-1);
errorlasso=zeros(1,length(parameters));
errorridge=zeros(1,length(parameters));
for i=1:sets    %loop over cross-validation for all data and get mean error for each holdout set
    train_x=newX(~logical(CV_blocks(:,i)),:);
    train_y=newy(~logical(CV_blocks(:,i)));
    test_x=newX(logical(CV_blocks(:,i)),:);
    test_y=newy(logical(CV_blocks(:,i)));
    indexcvallp=indexall(indexcvall~=i);%validation sets index of rest of training set
    for j=1:length(parameters)  %loop over number of parameters and get mean error for each parameter
        for k=1:(sets-1)   %calculate error of each holdout set
            indexcv_testp=indexcvallp(k);
            trainp_x=train_x(~logical(CV_pblocks(:,indexcv_testp)),:);
            trainp_y=train_y(~logical(CV_pblocks(:,indexcv_testp)));
         
            testp_x=train_x(logical(CV_blocks(:,indexcv_testp)),:);
            testp_y=train_y(logical(CV_pblocks(:,indexcv_testp)));
            B=lasso(trainp_x,trainp_y,parameters);
            errorpcvlasso(k)=sum(sign(testp_x*betap)~=test_y);
            beta_ridge=inv(trainp_x'*trainp_x+parameters(j)*eye(size(trainp_x,2)))*trainp_x'*trainp_y;
            errorpcvridge(k)=sum(sign(testp_x*betap)~=test_y);
        end
        errorlasso(j)=mean(errorpcvlasso);
        errorridge(j)=mean(errorpcvridge);
    end
    [sorterrorlasso,indexsortlasso]=sort(errorlasso);
    parameterlasso=parameters(indexsortlasso(1));
    [sorterrorridge,indexsortridge]=sort(errorridge);
    parameterridge=parameters(indexsortridge(1));
    beta_lassof=inv(train_x'*train_x+parameters(j)*eye(size(train_x,2)))*train_x'*train_y;
    erroralllasso(i)=sum(sign(test_x*beta)~=test_y);
    beta_ridgef=inv(train_x'*train_x+parameters(j)*eye(size(train_x,2)))*train_x'*train_y;
    errorallridge(i)=sum(sign(test_x*beta)~=test_y);
end

