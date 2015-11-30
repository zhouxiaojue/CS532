function [U,r]=orthogon(A)
q1=A(:,1)/norm(A(:,1));
U=[q1];
s=[];
for i=2:size(A,2)
    vj=A(:,i);
    for j=1:size(U,2)
        s(j)=U(:,j)'*A(:,i);
        vj=vj-s(j)*U(:,j);
    end
    if ~isempty(vj)
       U=[U,vj/norm(vj)];
    end 
end
r=size(U,2);
end

