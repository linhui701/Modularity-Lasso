function [indices, folds] = sparse_reg_fold_generation(n_sbj,n_folds)
% ------------------------------------------------------------------------%
% fold generation for cross-fold validation
% Input:
%   - num: total subjects number that need to be partitioned
%   - k:   fold number
% Output:
%   - fold: 1xk cell array, each cell stores the index of subjects for each
%           fold
% Author: Jingwen Yan
% Last update: 05/19/2019 by Linhui Xie
% ------------------------------------------------------------------------%
if nargin >= 3 
    sprintf('number and diagnosis info do not match');
    return;
end

if nargin < 2
    sprintf('Insufficient parameters!');
    return;
end

for i=1:n_folds
    folds{i} = [];
end

indices = zeros(n_sbj,1);
idx = find(indices==0);
n = n_sbj;

n_sbj = zeros(ceil(n/n_folds),n_folds);
n_sbj(1:n) = randperm(n); % random permutation
% n_sbj; 53 rows, 5 cols. Col num equal to num of fold
% n_sbj =
%    123    77   258    83   203
%     99   211   128    59    73
%    174   235   145   210     1
%      .     .     .     .     .
%      .     .     .     .     .
%      .     .     .     .     .
%     45    42   169   130     0
%    184   162   248    15     0
%    216    28   138   243     0

% reorder the n_sbj to cell array and remove the zero element;
for j=1:n_folds
    folds{j} = union(folds{j},idx(setdiff(n_sbj(:,j),0)));
end

for i = 1 : length(folds)
    indices(folds{i}) = i;
end
% indices: 262 rows 1 col. Each value stand for the the corresponding sub 
% is assigned to which fold
% indices =
% 
%      5
%      5
%      1
%      5
%      4

clear idx n i j;
