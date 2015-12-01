%%
%1
m = 1000;
n = 2;
b = zeros(m,1);
figure(1); 
subplot(121); hold on;
for i=1:m
    a = 2*rand(2,1)-1;
    A(i,:)=a';
    b(i) = sign(a(1)^2+a(2)^2-.5);
    if b(i)==1
        plot(a(1),a(2),'b.');
    else
        plot(a(1),a(2),'r.');
    end
end
axis('square')
title('training data')

subplot(122); hold on;
lambda=1e-5;
K=A*A';
aLS = (K+lambda*eye(size(K,1)))\b;
bLS = sign(K*aLS);

for i=1:m
    a = A(i,:);
    if bLS(i)==1
        plot(a(1),a(2),'b.');
    else
        plot(a(1),a(2),'r.');
    end
end
axis('square')
title('least squares')
%%
%2 gaussian kernel
m = 1000;
n = 2;
b = zeros(m,1);
figure(1); 
subplot(121); hold on;
for i=1:m
    a = 2*rand(2,1)-1;
    A(i,:)=a';
    b(i) = sign(a(1)^2+a(2)^2-.5);
    if b(i)==1
        plot(a(1),a(2),'b.');
    else
        plot(a(1),a(2),'r.');
    end
end
axis('square')
title('training data')

subplot(122); hold on;
lambda=1e-5;
for i=1:size(A,1)
    ai=A(i,:);
    for j=1:size(A,1)
        aj=A(j,:);
        K(i,j)=exp(-1/2*norm(ai-aj,1)^2);
    end
end
aLS = (K+lambda*eye(size(K,1)))\b;
bLS = sign(K*aLS);

for i=1:m
    a = A(i,:);
    if bLS(i)==1
        plot(a(1),a(2),'b.');
    else
        plot(a(1),a(2),'r.');
    end
end
axis('square')
title('gaussian kernel')
%%
%3polynomial kernel
m = 1000;
n = 2;
b = zeros(m,1);
figure(1); 
subplot(121); hold on;
for i=1:m
    a = 2*rand(2,1)-1;
    A(i,:)=a';
    b(i) = sign(a(1)^2+a(2)^2-.5);
    if b(i)==1
        plot(a(1),a(2),'b.');
    else
        plot(a(1),a(2),'r.');
    end
end
axis('square')
title('training data')

subplot(122); hold on;
lambda=1e-5;
for i=1:size(A,1)
    ai=A(i,:);
    for j=1:size(A,1)
        aj=A(j,:);
        K(i,j)=(ai*aj'+1)^2;
    end
end
aPL = (K+lambda*eye(size(K,1)))\b;
bPL = sign(K*aPL);

for i=1:m
    a = A(i,:);
    if bPL(i)==1
        plot(a(1),a(2),'b.');
    else
        plot(a(1),a(2),'r.');
    end
end
axis('square')
title('polynomial kernel')
%%
%4different training data sets with three types of classifiers
eLSall=[];
eGKall=[];
ePLall=[];
for k=1:100
    m = 1000;
    n = 2;
    b = zeros(m,1);
    for i=1:m
        a = 2*rand(2,1)-1;
        A(i,:)=a';
        b(i) = sign(a(1)^2+a(2)^2-.5);       
    end
    
    mt = 100;
    nt = 2;
    bt = zeros(mt,1);
    for i=1:mt
        at = 2*rand(2,1)-1;
        At(i,:)=at';
        bt(i) = sign(at(1)^2+at(2)^2-.5);       
    end
    
    %LS
    lambda=1e-5;
    K=A*A';
    aLS = (K+lambda*eye(size(K,1)))\b;
    eLSall(k)=sum(bt~=sign(At*A'*aLS));
    
    %Gaussian
    lambda=1e-5;
    for i=1:size(A,1)
        ai=A(i,:);
        for j=1:size(A,1)
            aj=A(j,:);
            K(i,j)=exp(-1/2*norm(ai-aj,1)^2);
        end
    end
    aGK = (K+lambda*eye(size(K,1)))\b;
    eGKall(k)=sum(bt~=sign(At*A'*aGK));
   
    %polynomial kernel
    lambda=1e-5;
    for i=1:size(A,1)
        ai=A(i,:);
        for j=1:size(A,1)
            aj=A(j,:);
            K(i,j)=(ai*aj'+1)^2;
        end
    end
    aPL = (K+lambda*eye(size(K,1)))\b;
    ePLall(k)=sum(bt~=sign(At*A'*aPL));
end

MSEPL=mean(ePLall);
MSEGK=mean(eGKall);
MSELS=mean(eLSall);
%m=10 MSEPL:50.17 MSEGK:50.07 MSELS:50.37
%m=100 MSEPL:49.94 MSEGK:49.58 MSELS:49.97
%m=1000 MSEPL:49.21 MSEGK:49.25 MSELS:49.67
%%
%5
eLSall=[];
eGKall=[];
ePLall=[];
for k=1:100
    m = 10;
    n = 2;
    b = zeros(m,1);
    for i=1:m
        a = 2*rand(2,1)-1;
        A(i,:)=a';
        b(i) = sign(a(1)^2+a(2)^2-.5);       
    end
    
    mt = 100;
    nt = 2;
    bt = zeros(mt,1);
    for i=1:mt
        at = 2*rand(2,1)-1;
        At(i,:)=at';
        bt(i) = sign(at(1)^2+at(2)^2-.5);       
    end
    
    %LS
    lambda=1e-5;
    K=A*A';
    aLS = (K+lambda*eye(size(K,1)))\b;
    eLSall(k)=sum(bt~=sign(At*A'*aLS));
    
    %Gaussian kernel svm
    aGK = svmtrain(A,b,'Kernel_Function','rbf');
    eGKall(k)=sum(svmclassify(aGK,At)~=bt);
   
    %polynomial kernel
    aPL=svmtrain(A,b,'Kernel_Function',@(u,v) (u*v'+1)^2);
    ePLall(k)=sum(svmclassify(aPL,At)~=bt);
end
MSEPL=mean(ePLall);
MSEGK=mean(eGKall);
MSELS=mean(eLSall);
%m=10 MSEPL:MSEGK: MSELS:
%m=100 MSEPL: MSEGK:49.58 MSELS:
%m=1000 MSEPL: MSEGK: MSELS:

%%
clear
%6-8
%transfer feet to inches
train=[70 71 73 82]';
class=[-1 -1 1 1]';
out=svmtrain(train,class);
predict=svmclassify(out,train);
%here the hinge loss cut the boundary between 6'1'' and 6'10''
%7
out.SupportVectors
%nonzero values are the last one, the second closest to zero is the third
%one with value -0.1826

%8
