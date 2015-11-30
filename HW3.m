%Question 1
A=[3 1;0 3;0 4];
b=[1;3;1];
beta=A*inv(A'*A)*A'*b;
% 
% beta =
% 
%     1.0000
%     1.5600
%     2.0800
clear


%Question 3

% function [U,r]=orthogon(A)
% q1=A(:,1)/norm(A(:,1));
% U=[q1];
% s=[];
% for i=2:size(A,2)
%     vj=A(:,i);
%     for j=1:size(U,2)
%         s(j)=U(:,j)'*A(:,i);
%         vj=vj-s(j)*U(:,j);
%     end
%     if ~isempty(vj)
%        U=[U,vj/norm(vj)];
%     end 
% end
% r=size(U,2);
% end

orthogon(A)
% 
%     1.0000         0
%          0    0.6000
%          0    0.8000

A1=[3 1 2;0 3 3;0 4 4;6 1 4];
A2=[1 1 2;0 3 3;0 4 4;3 1 4];

r1=size(orthogon(A1),2);
%r1=3 =rank(A1)
r2=size(orthogon(A2),2);
%r2=3~=rank(A2)

%Question 4
clear
load fisheriris.mat

%(a)
y=[repmat(1,50,1);repmat(0,50,1);repmat(-1,50,1)];
beta=inv(meas'*meas)*meas'*y;

predict=meas*beta;
% I would classify to closest label. for example, number that's in range
% -1/3 to -1 is last label. -1/3 to 1/3 is 0, 1/3 to 1 is 1.

%(b)
CVblocks=ones(150,1);
CVblocks([41:50,91:100,141:150],:)=0; %assign testing to 0
train_y=y(logical(CVblocks));
train_x=meas(logical(CVblocks),:);
test_y=y(~logical(CVblocks));
test_x=meas(~logical(CVblocks),:);

beta=inv(train_x'*train_x)*train_x'*train_y;
predict=test_x*beta;
predict(predict>=(1/3))=1;
predict(predict<=(-1/3))=-1;
predict(predict~=1&predict~=-1)=0;
error=sum(predict~=test_y);
MSE=sum(error)/30;

%(c)

CVblocks2=ones(150,39);
MSE2=[];
for j=1:39
    CVblocks2([(j+1):50,(51+j):100,(101+j):150],j)=0;
    train_y=y(logical(CVblocks2(:,j)));
    train_x=meas(logical(CVblocks2(:,j)),:);
    test_y=y(~logical(CVblocks2(:,j)));
    test_x=meas(~logical(CVblocks2(:,j)),:);
    beta=inv(train_x'*train_x)*train_x'*train_y;
    predict=test_x*beta;
    predict(predict>=(1/3))=1;
    predict(predict<=(-1/3))=-1;
    predict(predict~=1&predict~=-1)=0;
    error=sum(predict~=test_y);
    MSE2(j)=sum(error)/30;
end
MSE2(40)=MSE;
plot((1:40),MSE2)


%(d)
CVblocks3=ones(150,1);
CVblocks3([41:50,91:100,141:150],:)=0; %assign testing to 0
train_y=y(logical(CVblocks3));
train_x=meas(logical(CVblocks3),1:3);
test_y=y(~logical(CVblocks3));
test_x=meas(~logical(CVblocks3),1:3);

beta=inv(train_x'*train_x)*train_x'*train_y;
predict=test_x*beta;
predict(predict>=(1/3))=1;
predict(predict<=(-1/3))=-1;
predict(predict~=1&predict~=-1)=0;
error3=sum(predict~=test_y);
MSE3=error3/30
%(e)
scatter3(meas(:,1),meas(:,2),meas(:,3))
%yes 
%find the plane that fits to one set of data and another plane for another
%set 
index=kmeans(meas(:,1:3),3); %find the clusters the plane is the average plane between clusters


%(f)
%use the plane between two clusters from above to classify
% which means, the category belongs to 1 is set as label 1, category 2 as
% label 2, category 3 as label 3.


index=kmeans(meas(:,1:3),3);
y_2=[repmat(1,50,1);repmat(2,50,1);repmat(3,50,1)];

error3=sum(index~=y_2);
MSE3=error3/150

%MSE3=0.75
