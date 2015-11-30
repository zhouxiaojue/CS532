%Question#3
addpath /Users/kaeda/Documents/uw/2015Fall/CS532/HW2
load polydata.mat

%d=1;

A=ones(size(a,1),2);
A(:,2)=a;
p1=inv(A'*A)*A'*b;

%poly_fit1=polyfit(a,b,1);
a1=[min(a):(max(a)-min(a))/200:max(a)]; %create x for plotting polynomial fit
A1=ones(size(a1,2),2);
A1(:,2)=a1;
y1=A1*p1;
%y1=polyval(poly_fit1,a1);
plot(a,b,'o',a1,y1,'r')

A=ones(size(a,1),3);
A(:,2)=a;
A(:,3)=a.^2;
p1=inv(A'*A)*A'*b;
a1=[min(a):(max(a)-min(a))/200:max(a)]; %create x for plotting polynomial fit
A1=ones(size(a1,2),3);
A1(:,2)=a1;
A1(:,3)=a1.^2;
y2=A1*p1;
%poly_fit2=polyfit(a,b,2);
%y2=polyval(poly_fit2,a1);
plot(a,b,'o',a1,y2,'r')

A=ones(size(a,1),4);
A(:,2)=a;
A(:,3)=a.^2;
A(:,4)=a.^3;
p1=inv(A'*A)*A'*b;
a1=[min(a):(max(a)-min(a))/200:max(a)]; %create x for plotting polynomial fit
A1=ones(size(a1,2),4);
A1(:,2)=a1;
A1(:,3)=a1.^2;
A1(:,4)=a1.^3;
y3=A1*p1;
%poly_fit3=polyfit(a,b,3);
%y3=polyval(poly_fit3,a1);
plot(a,b,'o',a1,y3,'r') %why would this pair works??

clear
%Question#3
%a
A3=[25 0 1;20 1 2;40 1 6];
b3=[110;110;210];
x_hat=(A3'*A3)\(A3'*b3);

% x_hat =
% 
%     4.2500
%    17.5000
%     3.7500
% 

%b
x_star=[4;9;4];
fat=[1;1;1];
for i=1:3
  fat(i)=(b3(i)-(x_star(1)*A3(i,1))-(x_star(3)*A3(i,3)))/9;  
end  
%
% fat =
% 
%     0.6667
%     2.4444
%     2.8889


%c
A4=[25 15 10 0 1;20 12 8 1 2;40 30 10 1 6;30 15 15 0 3;35 20 15 2 4];
b4=[104;97;193;132;174];
x_hat2=inv(A4'*A4)*(A4'*b4);

%no we can't since A is not invertible. 
x_star2=[1;1;2;9;4];
b_2=A4*x_star2;


clear

%Question#6

%a
load face_emotion_data.mat
beta_hat=(X'*X)\(X'*y);
%
% beta_hat =
% 
%     0.9437
%     0.2137
%     0.2664
%    -0.3922
%    -0.0054
%    -0.0176
%    -0.1663
%    -0.0823
%    -0.1664

%b
%I would use these weights to multiply to new face images feature and the
%summed outcome is the happy if it's bigger than 0 and mad if it's negative

%c
%the first one since the weight is highest.

%d
% I would choose the [0.9437;-0.3922;0.2664] since they are the biggest
% weights.

%e
n=length(y);
%index=randperm(n);
%X2=X(index,:);
X2=X;
CVfolds=8;
ntest=n/CVfolds;
CVblocks=zeros(n,CVfolds);
j=1;
for i=1:CVfolds
    CVblocks((j:(j+ntest-1)),i)=1;
    j=j+ntest;
end
error=zeros(1,CVfolds);
beta=zeros(size(X2,2),CVfolds);
for i=1:CVfolds
    train=X2(~logical(CVblocks(:,i)),:);
    test=X2(logical(CVblocks(:,i)),:);
    test_y=y(logical(CVblocks(:,i)),:);
    beta(:,i)=inv(train'*train)*train'*y(~logical(CVblocks(:,i)));
    error(i)=(sum(sign(test*beta(:,i))~=test_y))/ntest;
end

MSE=mean(error)
%MSE=0.0469
clear all
%d
load face_emotion_data.mat
n=length(y);
index=randperm(n);
X2=X(:,[1 3 4]);
CVfolds=8;
ntest=n/CVfolds;
CVblocks=zeros(n,CVfolds);
j=1;
for i=1:CVfolds
    CVblocks((j:(j+ntest-1)),i)=1;
    j=j+ntest;
end
error=zeros(1,CVfolds);
beta=zeros(size(X2,2),CVfolds);
for i=1:CVfolds
    train=X2(~logical(CVblocks(:,i)),:);
    train_y=y(~logical(CVblocks(:,i)));
    test=X2(logical(CVblocks(:,i)),:);
    test_y=y(logical(CVblocks(:,i)));
    beta(:,i)=(train'*train)\train'*train_y;
    error(i)=(sum(sign(test*beta(:,i))~=test_y))/ntest;
end

MSE=mean(error)

%MSE=0.0781
