addpath /Users/kaeda/Documents/uw/2015Fall/CS532/HW2
load face_emotion_data.mat
%Problem 1
sets=8;
n=128/sets;
one_block=[ones(1,n),zeros(1,128-n)];
CV_blocks=[];
for i=1:sets
    CV_blocks=[CV_blocks circshift(one_block',(i-1)*n,1)];
end


onep_block=[ones(1,n),zeros(1,128-n-n)];
CV_pblocks=[];
for i=1:(sets-1)
    CV_pblocks=[CV_pblocks circshift(onep_block',(i-1)*n,1)];
end

%a
parameters=1:9;
error=zeros(1,sets);
errorpcv=zeros(1,sets-1);
errorp=zeros(1,length(parameters));
for i=1:sets    %loop over cross-validation for all data and get mean error for each holdout set
    train_x=X(~logical(CV_blocks(:,i)),:);
    train_y=y(~logical(CV_blocks(:,i)));
    test_x=X(logical(CV_blocks(:,i)),:);
    test_y=y(logical(CV_blocks(:,i)));
    for j=1:length(parameters)  %loop over number of parameters and get mean error for each parameter
        for k=1:(sets-1)   %calculate error of each holdout set
            trainp_x=train_x(~logical(CV_pblocks(:,k)),:);
            trainp_y=train_y(~logical(CV_pblocks(:,k)));
            testp_x=train_x(logical(CV_pblocks(:,k)),:);
            testp_y=train_y(logical(CV_pblocks(:,k)));
            [U,S,V]=svd(trainp_x);
            betap=V(:,1:j)*((S(1:j,1:j))\(U(:,1:j)'*trainp_y));
            errorpcv(k)=mean(sum(sign(testp_x*betap)~=test_y));
        end
        errorp(j)=mean(errorpcv);
    end
    [sorterrorp,indexsort]=sort(errorp);
    parameter=parameters(indexsort(1));
    [U,S,V]=svd(train_x);
    beta=V(:,1:parameter)*((S(1:parameter,1:parameter))\(U(:,1:parameter)'*train_y));
    error(i)=mean(sum(sign(test_x*beta)~=test_y));
end

MSE_SVD=mean(error);
%3.250 only with 2 singular values

%b
parameters=[0,1/2,2^0,2^1,2^2,2^3,2^4];
error=zeros(1,sets);
errorpcv=zeros(1,sets-1);
errorp=zeros(1,length(parameters));
for i=1:sets    %loop over cross-validation for all data and get mean error for each holdout set
    train_x=X(~logical(CV_blocks(:,i)),:);
    train_y=y(~logical(CV_blocks(:,i)));
    test_x=X(logical(CV_blocks(:,i)),:);
    test_y=y(logical(CV_blocks(:,i)));
    for j=1:length(parameters)  %loop over number of parameters and get mean error for each parameter
        for k=1:(sets-1)   %calculate error of each holdout set
            trainp_x=train_x(~logical(CV_pblocks(:,k)),:);
            trainp_y=train_y(~logical(CV_pblocks(:,k)));
            testp_x=train_x(logical(CV_pblocks(:,k)),:);
            testp_y=train_y(logical(CV_pblocks(:,k)));
            betap=inv(trainp_x'*trainp_x+parameters(j)*eye(size(trainp_x,2)))*trainp_x'*trainp_y;
            errorpcv(k)=mean(sum(sign(testp_x*betap)~=test_y));
        end
        errorp(j)=mean(errorpcv);
    end
    [sorterrorp,indexsort]=sort(errorp);
    parameter=parameters(indexsort(1));
    beta=inv(train_x'*train_x+parameters(j)*eye(size(train_x,2)))*train_x'*train_y;
    error(i)=mean(sum(sign(test_x*beta)~=test_y));
end

MSE_RLS=mean(error);
%0.75 with lambda=0, interestingly, the CV error for different lambda is
%the same for 0, 2^3, 2^4

%c

%%%%%%%%%
newX=[X X*randn(9,3)];
parameters=1:9;
error=zeros(1,sets);
errorpcv=zeros(1,sets-1);
errorp=zeros(1,length(parameters));
for i=1:sets    %loop over cross-validation for all data and get mean error for each holdout set
    train_x=newX(~logical(CV_blocks(:,i)),:);
    train_y=y(~logical(CV_blocks(:,i)));
    test_x=newX(logical(CV_blocks(:,i)),:);
    test_y=y(logical(CV_blocks(:,i)));
    for j=1:length(parameters)  %loop over number of parameters and get mean error for each parameter
        for k=1:(sets-1)   %calculate error of each holdout set
            trainp_x=train_x(~logical(CV_pblocks(:,k)),:);
            trainp_y=train_y(~logical(CV_pblocks(:,k)));
            testp_x=train_x(logical(CV_pblocks(:,k)),:);
            testp_y=train_y(logical(CV_pblocks(:,k)));
            [U,S,V]=svd(trainp_x);
            betap=V(:,1:j)*((S(1:j,1:j))\(U(:,1:j)'*trainp_y));
            errorpcv(k)=mean(sum(sign(testp_x*betap)~=test_y));
        end
        errorp(j)=mean(errorpcv);
    end
    [sorterrorp,indexsort]=sort(errorp);
    parameter=parameters(indexsort(1));
    [U,S,V]=svd(train_x);
    beta=V(:,1:parameter)*((S(1:parameter,1:parameter))\(U(:,1:parameter)'*train_y));
    error(i)=mean(sum(sign(test_x*beta)~=test_y));
end

MSE_SVD2=mean(error);
%2.625


%%%%%%%%%%%%%%%RLS
parameters=[0,1/2,2^0,2^1,2^2,2^3,2^4];
error=zeros(1,sets);
errorpcv=zeros(1,sets-1);
errorp=zeros(1,length(parameters));
for i=1:sets    %loop over cross-validation for all data and get mean error for each holdout set
    train_x=newX(~logical(CV_blocks(:,i)),:);
    train_y=y(~logical(CV_blocks(:,i)));
    test_x=newX(logical(CV_blocks(:,i)),:);
    test_y=y(logical(CV_blocks(:,i)));
    for j=1:length(parameters)  %loop over number of parameters and get mean error for each parameter
        for k=1:(sets-1)   %calculate error of each holdout set
            trainp_x=train_x(~logical(CV_pblocks(:,k)),:);
            trainp_y=train_y(~logical(CV_pblocks(:,k)));
            testp_x=train_x(logical(CV_pblocks(:,k)),:);
            testp_y=train_y(logical(CV_pblocks(:,k)));
            betap=inv(trainp_x'*trainp_x+parameters(j)*eye(size(trainp_x,2)))*trainp_x'*trainp_y;
            errorpcv(k)=mean(sum(sign(testp_x*betap)~=test_y));
        end
        errorp(j)=mean(errorpcv);
    end
    [sorterrorp,indexsort]=sort(errorp);
    parameter=parameters(indexsort(1));
    beta=inv(train_x'*train_x+parameters(j)*eye(size(train_x,2)))*train_x'*train_y;
    error(i)=mean(sum(sign(test_x*beta)~=test_y));
end

MSE_RLS2=mean(error);

%0.875


%%
%2
%a

% deblurring
clear 
close all
n = 500;
k = 30;
sigma = 0.01;

% generate random piecewise constant signal
x = zeros(n,1);
x(1) = randn;
for i=2:n
    if (rand < .95)
        x(i) = x(i-1);
    else
        x(i) = randn;
    end
end
    
% generate k-point averaging function
h = ones(1,k)/k;

% make A matrix for blurring 
m = n+k-1;
for i=1:m
    if i<=k
        A(i,1:i) = h(1:i);
    else
        A(i,i-k+1:i) = h;
    end
end
A = A(:,1:n);

% blurred signal + noise
b = A*x+sigma*randn(m,1);


%%%%
%OLS
beta_ols=inv(A'*A)*A'*b;

%Model with CV
sets=10;
n=(529+1)/sets;
one_block=[ones(1,n),zeros(1,529+1-n)];
CV_blocks=[];
for i=1:sets
    CV_blocks=[CV_blocks circshift(one_block',(i-1)*n,1)];
end


onep_block=[ones(1,n),zeros(1,529+1-n-n)];
CV_pblocks=[];
for i=1:(sets-1)
    CV_pblocks=[CV_pblocks circshift(onep_block',(i-1)*n,1)];
end


b(530)=b(529);
A(530,:)=A(529,:);
%%%%SVD
parameters=1:9;
error=zeros(1,sets);
errorpcv=zeros(1,sets-1);
errorp=zeros(1,length(parameters));
for i=1:sets    %loop over cross-validation for all data and get mean error for each holdout set
    train_x=A(~logical(CV_blocks(:,i)),:);
    train_y=b(~logical(CV_blocks(:,i)));
    test_x=A(logical(CV_blocks(:,i)),:);
    test_y=b(logical(CV_blocks(:,i)));
    for j=1:length(parameters)  %loop over number of parameters and get mean error for each parameter
        for k=1:(sets-1)   %calculate error of each holdout set
            trainp_x=train_x(~logical(CV_pblocks(:,k)),:);
            trainp_y=train_y(~logical(CV_pblocks(:,k)));
            testp_x=train_x(logical(CV_pblocks(:,k)),:);
            testp_y=train_y(logical(CV_pblocks(:,k)));
            [U,S,V]=svd(trainp_x);
            betap=V(:,1:j)*((S(1:j,1:j))\(U(:,1:j)'*trainp_y));
            errorpcv(k)=mean(testp_x*betap-test_y);
        end
        errorp(j)=mean(errorpcv);
    end
    [sorterrorp,indexsort]=sort(errorp);
    parameter=parameters(indexsort(1));
    [U,S,V]=svd(train_x);
    beta_svd=V(:,1:parameter)*((S(1:parameter,1:parameter))\(U(:,1:parameter)'*train_y));
    error(i)=mean(test_x*beta_svd-test_y);
end


%%%%%%%%%%%%%%%RLS
parameters=[0,1/2,2^0,2^1,2^2,2^3,2^4];
error=zeros(1,sets);
errorpcv=zeros(1,sets-1);
errorp=zeros(1,length(parameters));
for i=1:sets    %loop over cross-validation for all data and get mean error for each holdout set
    train_x=A(~logical(CV_blocks(:,i)),:);
    train_y=b(~logical(CV_blocks(:,i)));
    test_x=A(logical(CV_blocks(:,i)),:);
    test_y=b(logical(CV_blocks(:,i)));
    for j=1:length(parameters)  %loop over number of parameters and get mean error for each parameter
        for k=1:(sets-1)   %calculate error of each holdout set
            trainp_x=train_x(~logical(CV_pblocks(:,k)),:);
            trainp_y=train_y(~logical(CV_pblocks(:,k)));
            testp_x=train_x(logical(CV_pblocks(:,k)),:);
            testp_y=train_y(logical(CV_pblocks(:,k)));
            betap=inv(trainp_x'*trainp_x+parameters(j)*eye(size(trainp_x,2)))*trainp_x'*trainp_y;
            errorpcv(k)=mean(testp_x*betap-test_y);
        end
        errorp(j)=mean(errorpcv);
    end
    [sorterrorp,indexsort]=sort(errorp);
    parameter=parameters(indexsort(1));
    beta_ridge=inv(train_x'*train_x+parameters(j)*eye(size(train_x,2)))*train_x'*train_y;
    error(i)=mean(test_x*beta_ridge-test_y);
end

figure(1)
subplot(211)
plot(x)
hold on
a=plot(beta_ols,'yellow')
a1=plot(beta_svd,'green')
a2=plot(beta_ridge,'red')
hold off
legend([a,a1,a2],'OLS','TSVD','RIDGE')
t=title('signal')
set(gca,'Fontsize',16)
set(t,'Fontsize',16)
subplot(212)
plot(b(1:n))
hold on
a=plot(beta_ols,'yellow')
a1=plot(beta_svd,'green')
a2=plot(beta_ridge,'red')
hold off
legend([a,a1,a2],'OLS','TSVD','RIDGE')
axis('tight')
t=title('blurred and noisy version')
set(t,'Fontsize',16)
set(gca,'Fontsize',16)

%%
%2b when k is quite big, it's interesting to know the 
clear 
close all
n = 500;
k = 500;
sigma = 0.01;

% generate random piecewise constant signal
x = zeros(n,1);
x(1) = randn;
for i=2:n
    if (rand < .95)
        x(i) = x(i-1);
    else
        x(i) = randn;
    end
end
    
% generate k-point averaging function
h = ones(1,k)/k;

% make A matrix for blurring 
m = n+k-1;
for i=1:m
    if i<=k
        A(i,1:i) = h(1:i);
    else
        A(i,i-k+1:i) = h;
    end
end
A = A(:,1:n);

% blurred signal + noise
b = A*x+sigma*randn(m,1);


%%%%
%OLS
beta_ols=inv(A'*A)*A'*b;

%Model with CV
sets=10;
n=(529+1)/sets;
one_block=[ones(1,n),zeros(1,529+1-n)];
CV_blocks=[];
for i=1:sets
    CV_blocks=[CV_blocks circshift(one_block',(i-1)*n,1)];
end


onep_block=[ones(1,n),zeros(1,529+1-n-n)];
CV_pblocks=[];
for i=1:(sets-1)
    CV_pblocks=[CV_pblocks circshift(onep_block',(i-1)*n,1)];
end


b(530)=b(529);
A(530,:)=A(529,:);
%%%%SVD
parameters=1:9;
error=zeros(1,sets);
errorpcv=zeros(1,sets-1);
errorp=zeros(1,length(parameters));
for i=1:sets    %loop over cross-validation for all data and get mean error for each holdout set
    train_x=A(~logical(CV_blocks(:,i)),:);
    train_y=b(~logical(CV_blocks(:,i)));
    test_x=A(logical(CV_blocks(:,i)),:);
    test_y=b(logical(CV_blocks(:,i)));
    for j=1:length(parameters)  %loop over number of parameters and get mean error for each parameter
        for k=1:(sets-1)   %calculate error of each holdout set
            trainp_x=train_x(~logical(CV_pblocks(:,k)),:);
            trainp_y=train_y(~logical(CV_pblocks(:,k)));
            testp_x=train_x(logical(CV_pblocks(:,k)),:);
            testp_y=train_y(logical(CV_pblocks(:,k)));
            [U,S,V]=svd(trainp_x);
            betap=V(:,1:j)*((S(1:j,1:j))\(U(:,1:j)'*trainp_y));
            errorpcv(k)=mean(testp_x*betap-test_y);
        end
        errorp(j)=mean(errorpcv);
    end
    [sorterrorp,indexsort]=sort(errorp);
    parameter=parameters(indexsort(1));
    [U,S,V]=svd(train_x);
    beta_svd=V(:,1:parameter)*((S(1:parameter,1:parameter))\(U(:,1:parameter)'*train_y));
    error(i)=mean(test_x*beta_svd-test_y);
end


%%%%%%%%%%%%%%%RLS
parameters=[0,1/2,2^0,2^1,2^2,2^3,2^4];
error=zeros(1,sets);
errorpcv=zeros(1,sets-1);
errorp=zeros(1,length(parameters));
for i=1:sets    %loop over cross-validation for all data and get mean error for each holdout set
    train_x=A(~logical(CV_blocks(:,i)),:);
    train_y=b(~logical(CV_blocks(:,i)));
    test_x=A(logical(CV_blocks(:,i)),:);
    test_y=b(logical(CV_blocks(:,i)));
    for j=1:length(parameters)  %loop over number of parameters and get mean error for each parameter
        for k=1:(sets-1)   %calculate error of each holdout set
            trainp_x=train_x(~logical(CV_pblocks(:,k)),:);
            trainp_y=train_y(~logical(CV_pblocks(:,k)));
            testp_x=train_x(logical(CV_pblocks(:,k)),:);
            testp_y=train_y(logical(CV_pblocks(:,k)));
            betap=inv(trainp_x'*trainp_x+parameters(j)*eye(size(trainp_x,2)))*trainp_x'*trainp_y;
            errorpcv(k)=mean(testp_x*betap-test_y);
        end
        errorp(j)=mean(errorpcv);
    end
    [sorterrorp,indexsort]=sort(errorp);
    parameter=parameters(indexsort(1));
    beta_ridge=inv(train_x'*train_x+parameters(j)*eye(size(train_x,2)))*train_x'*train_y;
    error(i)=mean(test_x*beta_ridge-test_y);
end

figure(1)
subplot(211)
plot(x)
hold on
a=plot(beta_ols,'yellow')
a1=plot(beta_svd,'green')
a2=plot(beta_ridge,'red')
hold off
legend([a,a1,a2],'OLS','TSVD','RIDGE')
t=title('signal')
set(gca,'Fontsize',16)
set(t,'Fontsize',16)
subplot(212)
plot(b(1:n))
hold on
a=plot(beta_ols,'yellow')
a1=plot(beta_svd,'green')
a2=plot(beta_ridge,'red')
hold off
legend([a,a1,a2],'OLS','TSVD','RIDGE')
axis('tight')
t=title('blurred and noisy version')
set(t,'Fontsize',16)
set(gca,'Fontsize',16)
%% when sigma is quite big
clear 
close all
n = 500;
k = 50;
sigma = 0.01;

% generate random piecewise constant signal
x = zeros(n,1);
x(1) = randn;
for i=2:n
    if (rand < .95)
        x(i) = x(i-1);
    else
        x(i) = randn;
    end
end
    
% generate k-point averaging function
h = ones(1,k)/k;

% make A matrix for blurring 
m = n+k-1;
for i=1:m
    if i<=k
        A(i,1:i) = h(1:i);
    else
        A(i,i-k+1:i) = h;
    end
end
A = A(:,1:n);

% blurred signal + noise
b = A*x+sigma*randn(m,1);

%%%%
%OLS
beta_ols=inv(A'*A)*A'*b;

%Model with CV
sets=10;
n=(529+1)/sets;
one_block=[ones(1,n),zeros(1,size(b,2)+1-n)];
CV_blocks=[];
for i=1:sets
    CV_blocks=[CV_blocks circshift(one_block',(i-1)*n,1)];
end


onep_block=[ones(1,n),zeros(1,529+1-n-n)];
CV_pblocks=[];
for i=1:(sets-1)
    CV_pblocks=[CV_pblocks circshift(onep_block',(i-1)*n,1)];
end


b(530)=b(529);
A(530,:)=A(529,:);
%%%%SVD
parameters=1:9;
error=zeros(1,sets);
errorpcv=zeros(1,sets-1);
errorp=zeros(1,length(parameters));
for i=1:sets    %loop over cross-validation for all data and get mean error for each holdout set
    train_x=A(~logical(CV_blocks(:,i)),:);
    train_y=b(~logical(CV_blocks(:,i)));
    test_x=A(logical(CV_blocks(:,i)),:);
    test_y=b(logical(CV_blocks(:,i)));
    for j=1:length(parameters)  %loop over number of parameters and get mean error for each parameter
        for k=1:(sets-1)   %calculate error of each holdout set
            trainp_x=train_x(~logical(CV_pblocks(:,k)),:);
            trainp_y=train_y(~logical(CV_pblocks(:,k)));
            testp_x=train_x(logical(CV_pblocks(:,k)),:);
            testp_y=train_y(logical(CV_pblocks(:,k)));
            [U,S,V]=svd(trainp_x);
            betap=V(:,1:j)*((S(1:j,1:j))\(U(:,1:j)'*trainp_y));
            errorpcv(k)=mean(testp_x*betap-test_y);
        end
        errorp(j)=mean(errorpcv);
    end
    [sorterrorp,indexsort]=sort(errorp);
    parameter=parameters(indexsort(1));
    [U,S,V]=svd(train_x);
    beta_svd=V(:,1:parameter)*((S(1:parameter,1:parameter))\(U(:,1:parameter)'*train_y));
    error(i)=mean(test_x*beta_svd-test_y);
end


%%%%%%%%%%%%%%%RLS
parameters=[0,1/2,2^0,2^1,2^2,2^3,2^4];
error=zeros(1,sets);
errorpcv=zeros(1,sets-1);
errorp=zeros(1,length(parameters));
for i=1:sets    %loop over cross-validation for all data and get mean error for each holdout set
    train_x=A(~logical(CV_blocks(:,i)),:);
    train_y=b(~logical(CV_blocks(:,i)));
    test_x=A(logical(CV_blocks(:,i)),:);
    test_y=b(logical(CV_blocks(:,i)));
    for j=1:length(parameters)  %loop over number of parameters and get mean error for each parameter
        for k=1:(sets-1)   %calculate error of each holdout set
            trainp_x=train_x(~logical(CV_pblocks(:,k)),:);
            trainp_y=train_y(~logical(CV_pblocks(:,k)));
            testp_x=train_x(logical(CV_pblocks(:,k)),:);
            testp_y=train_y(logical(CV_pblocks(:,k)));
            betap=inv(trainp_x'*trainp_x+parameters(j)*eye(size(trainp_x,2)))*trainp_x'*trainp_y;
            errorpcv(k)=mean(testp_x*betap-test_y);
        end
        errorp(j)=mean(errorpcv);
    end
    [sorterrorp,indexsort]=sort(errorp);
    parameter=parameters(indexsort(1));
    beta_ridge=inv(train_x'*train_x+parameters(j)*eye(size(train_x,2)))*train_x'*train_y;
    error(i)=mean(test_x*beta_ridge-test_y);
end

figure(1)
subplot(211)
plot(x)
hold on
a=plot(beta_ols,'yellow')
a1=plot(beta_svd,'green')
a2=plot(beta_ridge,'red')
hold off
legend([a,a1,a2],'OLS','TSVD','RIDGE')
t=title('signal')
set(gca,'Fontsize',16)
set(t,'Fontsize',16)
subplot(212)
plot(b(1:n))
hold on
a=plot(beta_ols,'yellow')
a1=plot(beta_svd,'green')
a2=plot(beta_ridge,'red')
hold off
legend([a,a1,a2],'OLS','TSVD','RIDGE')
axis('tight')
t=title('blurred and noisy version')
set(t,'Fontsize',16)
set(gca,'Fontsize',16)