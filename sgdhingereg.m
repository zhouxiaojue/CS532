function w=sgdhingereg(X,Y)
%%%%%%%%%%%%
%Y is m by 1 response, X is m by n
    iterations=2e4;
    wold=zeros(size(X,2),1);
    tol=1e-2;
    stepsize=3e-3;
    lambda=0.1;
    for i=1:iterations
        
        if Y'*X*wold<1
            w=wold+stepsize*(X'*Y+2*lambda*[1 1 0]');
        else
            w=wold-stepsize*2*lambda*[1 1 0]';
        end
        if norm(w-wold,1)<tol
            break
        end
        
        wold=w;
    end
end