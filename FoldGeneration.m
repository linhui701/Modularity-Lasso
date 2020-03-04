function [indices, fold] = FoldGeneration(num,k,DX)
% ------------------------------------------------------------------------%
% fold generation for cross-fold validation
% ------------------------------------------------------------------------%
% Input:
%   - num: total subjects number that need to be partitioned
%   - k:   fold number
%   - DX:  diagnosis of all the subjects, its length should match with "num"
% Output:
%   - fold: 1xk cell array, each cell stores the index of subjects for each
%           fold
% Author: Jingwen Yan
% Last update: 10/09/2012
% ------------------------------------------------------------------------%
if nargin == 3 && length(DX) ~= num
    print('number and diagnosis info do not match');
    return;
end

if nargin < 2
    print('Insufficient parameters!');
    return;
end

if nargin == 2
    DX = ones(num,1);
end
for i=1:k
    fold{i} = [];
end

indices = zeros(length(DX),1);
uni_DX = unique(DX);
for i = 1 : length(uni_DX)
    
    idx = find(DX == uni_DX(i));
    n = length(idx);
    
    num = zeros(ceil(n/k),k);
    num(1:n) = randperm(n); % random permutation
    for j=1:k
        fold{j} = union(fold{j},idx(setdiff(num(:,j),0)));
    end
end

for i = 1 : length(fold)
    
    indices(fold{i}) = i;
    
end


return;