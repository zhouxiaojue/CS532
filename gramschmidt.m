function Q = gramschmidt( A )
% FUNCTION computes orthonormal basis from independent vectors in A.
%
%   Q = gramschmidt( A );
%
% INPUT :
%   A   - a matrix with *INDEPENDENT* vectors in columns
%         (e.g. A = rand(3) will produce one)
%
% OUTPUT :
%   Q   - a matrix with orthonomal basis of A
%
% NOTE :
%   In practice use the qr() function!
%
% Vladislavs DOVGALECS
%

% The vectors in A are independent *BUT NOT YET* orthonormal. Check A'*A.
% If it is orthonormal, you should get strictly an identity matrix.

% number of vectors
n = size( A, 2 );

% initialize output
Q = zeros( n );

% turn every independent vector into a basis vector
% (1) jth basis vector will be perpendicular to 1..j-1 previous found basis
% (2) will be of length 1 (norm will be equal to 1)
for j = 1 : n
    
    % pick the jth independent vector
    u = A( :, j );
    
    % special case for j = 1: we will not run the following for loop. Will
    % just normalize it and put as the first found basis. There are no
    % previous basis to make orthogonal to.
    
    % remove from raw "u" all components spanned on 1..j-1 bases, their
    % contributions will be removed
    % ==> this effectively makes jth independent vector orthogonal to all
    % previous bases found in the previous steps.
    % ==> enforcing orthogonality principle here. Not orthonormality yet.
    for i = 1 : j - 1
        u = u - proj( Q(:,i), A(:,j) );
    end
    
    % normalize it to length of 1 and store it
    Q(:,j) = u ./ norm( u );
    
end

end

% projects a vector "a" on a direction "e"
function p = proj( e, a )

% project "a" onto "e": (e' * a) / (e' * e) is the length (!) of "a" on "e"
% multiplication by "e" is necessary to output the resulting vector which is
% (colinear with "e")
p = (e' * a) / (e' * e) .* e;

end
