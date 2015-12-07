addpath \\wfs1\users$\xiaojue\Downloads\CS532project\SMNI_CMI_TRAIN_features
load eeg_labels4.mat
y=aaa;
load entropy.mat
load features4_noentropy.mat

figure;
hold on
for i=1:length(features)
    subplot(211);
    plot(features(1:10,i))
    hold on
    subplot(212);
    plot(features(11:20,i))
    hold on
end
hold off

%%
%lasso
y1=y;
y1(11:20)=zeros(10,1)';
[Beta info]=lasso(features,y,'CV',5);
lassoPlot(Beta,info)
plot(info.Lambda,info.MSE)
xlabel('l1 penalty')
ylabel('MSE')
%choose 0.15

[Beta1 info1]=lasso(features,y,'CV',5,'Lambda',0.15);
predictLasso=features*Beta1;
subplot(211);
plot(y)
subplot(212);
plot(predictLasso)

index=randperm(length(y));
X=features(index,:);
y1=y(index,:);
r=4;
CVblocks=[ones(1,r) zeros(1,20-r)]';
first=[ones(1,r) zeros(1,20-r)]';
for i=1:r
    CVblocks=[CVblocks circshift(first,i*r)];
end
error=[];
for i=1:(20/r)
    xtrain=X(~CVblocks(:,i),:);
    ytrain=y1(~CVblocks(:,i),:);
    xtest=X(logical(CVblocks(:,i)),:);
    ytest=y1(logical(CVblocks(:,i)),:);
    [Beta1 info]=lasso(features,y,'Lambda',0.15);
    predictlasso=xtest*Beta1;
    error(i)=mean(ytest~=sign(predictlasso));
end
mean(error)

%without crossvalidation
[beta fit]=lasso(features,y,'Lambda',0.15);
predictlasso=features*beta;
mean(y~=sign(predictlasso))


[BetaRidge infoRidge]=lasso(features,y,'CV',4,'Alpha',0.01);
plot(infoRidge.Lambda,infoRidge.MSE)
xlabel('l1 penalty')
ylabel('MSE')

[BetaRidge1 infoRidge1]=lasso(features,y,'CV',4,'Alpha',0.01,'Lambda',2); %try 2/13/50 lambda bigger than 2 the same
predictRidge=features*BetaRidge1;
subplot(211);
plot(y)
subplot(212);
plot(predictLasso)

index=randperm(length(y));
X=features(index,:);
y1=y(index,:);
CVblocks=[1 zeros(1,20-1)]';
first=[1 zeros(1,20-1)]';
for i=1:19
    CVblocks=[CVblocks circshift(first,i)];
end
error=[];
for i=1:20
    xtrain=X(~CVblocks(:,i),:);
    ytrain=y1(~CVblocks(:,i),:);
    xtest=X(logical(CVblocks(:,i)),:);
    ytest=y1(logical(CVblocks(:,i)),:);
    [Beta1 info]=lasso(features,y,'Alpha',0.01,'Lambda',2);
    predictlasso=xtest*Beta1;
    error(i)=mean(ytest~=sign(predictlasso));
end
mean(error)

%without crossvalidation
[beta fit]=lasso(features,y,'Lambda',2,'Alpha',0.01);
predictridge=features*beta;
mean(y~=sign(predictridge))

[Beta1 info1]=lasso(features,y,'CV',4,'Lambda',);
%%
%SVD
[UA SA VA]=svd(features(1:10,:),'econ');
%2nd one is the most important feature
[UC SC VC]=svd(features(11:20,:),'econ');
%eighth one is most dominant 

%wt(:,i) = V*diag( diag(S)./(diag(S).ˆ2 + lambda) )*(U’*yt);


%%
%KNN

%without crossvalidation
outknn=fitcknn(features,y);
predictknn=predict(outknn,features);
mean(y~=predictknn)

index=randperm(length(y));
X=features(index,:);
y1=y(index,:);
r=4;
CVblocks=[ones(1,r) zeros(1,20-r)]';
first=[ones(1,r) zeros(1,20-r)]';
for i=1:r
    CVblocks=[CVblocks circshift(first,i*r)];
end
error=[];
for i=1:5
    xtrain=X(~CVblocks(:,i),:);
    ytrain=y1(~CVblocks(:,i),:);
    xtest=X(logical(CVblocks(:,i)),:);
    ytest=y1(logical(CVblocks(:,i)),:);
    outknn=fitcknn(xtrain,ytrain);
    predictknn=predict(outknn,xtest);
    error(i)=mean(ytest~=predictknn);
end
mean(error)
%SVM

index=randperm(length(y));
X=features(index,:);
y1=y(index,:);
r=4;
CVblocks=[ones(1,r) zeros(1,20-r)]';
first=[ones(1,r) zeros(1,20-r)]';
for i=1:r
    CVblocks=[CVblocks circshift(first,i*r)];
end
error=[];
for i=1:20
    xtrain=X(~CVblocks(:,i),:);
    ytrain=y1(~CVblocks(:,i),:);
    xtest=X(logical(CVblocks(:,i)),:);
    ytest=y1(logical(CVblocks(:,i)),:);
    outsvm=svmtrain(xtrain,ytrain);
    predictsvm=svmclassify(outsvm,xtest);
    error(i)=mean(ytest~=predictsvm);
end
mean(error)

%without crossvalidation
outcsvm=svmtrain(features,y);
predictsvm=svmclassify(outsvm,features);
mean(y~=predictsvm)


