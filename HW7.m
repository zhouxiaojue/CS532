addpath /Users/kaeda/Documents/uw/2015Fall/CS532/HW7
n=1;
w1=repmat(3,n,1);
w2=4;
m=10;
X=[];
Y=[];
for i=1:m
    x=rand(n,1);
    X(i,:)=x;
    sigma=randn(1);
    Y(i,:)=w1'*x+w2+sigma;
end
X=[ones(m,1) X];

%b
w_l1=sgdi(X,Y);
w_ols=(X'*X)\X'*Y;


plot(X(:,2),Y,'.')
hold on
xplot=linspace(min(X(:,2)),max(X(:,2)),2);
yplot_l1=w_l1(1)+w_l1(2)*xplot;
yplot_ols=w_ols(1)+w_ols(2)*xplot;

a=plot(xplot,yplot_l1,'-');
b=plot(xplot,yplot_ols,'-');
legend([a b],'LS line fit','l1 loss')
% ols is closer to the truth
%c
w1=repmat(3,n,1);
w2=4;
m=10;
X=[];
Y=[];
for i=1:m
    x=rand(n,1);
    X(i,:)=x;
    sigma=laprnd(1,1);
    Y(i,:)=w1'*x+w2+sigma;
end
X=[ones(m,1) X];

w_l1=sgdi(X,Y);
w_ols=(X'*X)\X'*Y;


plot(X(:,2),Y,'.')
hold on
xplot=linspace(min(X(:,2)),max(X(:,2)),2);
yplot_l1=w_l1(1)+w_l1(2)*xplot;
yplot_ols=w_ols(1)+w_ols(2)*xplot;

a=plot(xplot,yplot_l1,'-');
b=plot(xplot,yplot_ols,'-');
legend([a b],'LS line fit','l1 loss')
%now l1 is closer to the truth

%%
%question 4
clear
load fisheriris.mat

%1
X=meas(51:150,3:4);
X=[X ones(100,1)];
y=[ones(1,50) repmat(-1,1,50)]';
beta=(X'*X)\X'*y;

plotx=linspace(2,8,50);
ploty=-beta(3)/beta(2)-(beta(1)/beta(2))*plotx;

a=plot(meas(51:100,3),meas(51:100,4),'.');
axis([2 8 0 3])
hold on
b=plot(meas(101:150,3),meas(101:150,4),'.');
c=plot(plotx,ploty,'-');
xlabel('feature 3 (petal length)')
ylabel('feature 4 (petal width)')
legend([a,b,c],'versicolor','virginica','linear classification')

%b
w_SVM=sgdhingereg(X,y);
ploty_svm=-w_SVM(3)/w_SVM(2)-(w_SVM(1)/w_SVM(2))*plotx;
a=plot(meas(51:100,3),meas(51:100,4),'.');
axis([2 8 0 3])
hold on
b=plot(meas(101:150,3),meas(101:150,4),'.');
c=plot(plotx,ploty,'-');
d=plot(plotx,ploty_svm,'-');
xlabel('feature 3 (petal length)')
ylabel('feature 4 (petal width)')
legend([a,b,c,d],'versicolor','virginica','linear classification','SVM classifier')

%c
iterations=2e4;
wold=zeros(size(X,2),1);
tol=1e-4;
stepsize=3e-3;
lambda=0.1;
wall=[];
Y=y;
for i=1:iterations
    if Y'*X*wold<1
        w=wold+stepsize*(X'*Y+2*lambda*[1 1 0]');
    else
        w=wold-stepsize*2*lambda*[1 1 0]';
    end
    if norm(w-wold,1)<tol
        break
    end
    wall=[wall w];
    wold=w;
end
a=plot(1:iterations,wall(1,:));hold on
b=plot(1:iterations,wall(2,:));
c=plot(1:iterations,wall(3,:));
legend([a,b,c],'w1','w2','w3')