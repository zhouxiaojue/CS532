function w=sgdi(X,Y)
%%%%%%%%%%%%
%Y is m by 1 response, X is m by n
    iterations=1e5;
    wold=zeros(size(X,2),1);
    tol=1e-2;
    index=randperm(size(Y,1));
    for i=1:iterations
        if mod(i,size(Y,1))==0%index of 10th number is 10th in the indexlist
            y=Y(index(size(Y,1)),:);
            x=X(index(size(Y,1)),:);
        else
            y=Y(index(mod(i,size(Y,1))),:); %index of other number is the remainder
            x=X(index(mod(i,size(Y,1))),:);
        end
        
        if y-x*wold>0
            w=wold+(1/sqrt(i))*x';
        else
            w=wold-(1/sqrt(i))*x';
        end
        
        if norm(w-wold,1)<tol
            break
        end
        
        wold=w;
        if mod(i,10)==0
            index=randperm(size(Y,1));
        end
    end
end