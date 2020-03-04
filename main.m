clc;clear all;
% ------------------------------------------------------------------------%
% Run this example to see how to use
% ------------------------------------------------------------------------%
% Author: Linhui Xie, linhxie@iu.edu, xie215@purdue.edu
% Date created: May-10-2019
% Update: Jan-22-2020
% @Indiana University School of Medicine.
% @Purdue University Electrical and Computer Engineering.
% ------------------------------------------------------------------------%

% load ROSMAP data sets
load('../../ROSMAP_data/Regression.mat')
% contain data matrix x_adjust, cognitive performance vector y.
% contain modularity matrix B_max_abs_eig, modularity matrix L.

X=x_adjust;
Y=y;
% ------------------------------------------------------------------------%
for i = 1:929
    [row,col]=find(isnan(x(:,i))==1);
    col_mean = nanmean(x(:,i));
    for j = 1:length(row)
       X(row(j),i) =  col_mean;
    end
end
% ------------------------------------------------------------------------%

% set parameters for first round, should be tuned before running.
para.alpha=[10^-6 10^-5 10^-4 10^-3 10^-2 10^-1 10^0];
para.gamma=[10^-6 10^-5 10^-4 10^-3 10^-2 10^-1 ];

para.max_iter = 50;

% set CrossValidation numbers
n_folds = 5;
inner_folds=[];

% get normalization of X and Y.
X = getNormalization(X);
Y = getNormalization(Y);

flag_abs = 0;     % 0: without absolute constraint
                  % 1: with absolute constraint
flag_penalty = 2; % 0: modularity
                  % 1: laplacian
                  % 2: lasso
                  % 3: elastic net
                  % 4: lassoglm lasso
                  
if flag_penalty == 0
    B = B_max_abs_eig;
elseif flag_penalty == 1
    B = L;
end

[W, reg_loss, corr_XY, optimalParas] = sparse_reg_nested_CV(X, Y, B, para, n_folds, inner_folds, flag_abs, flag_penalty);

save('laplacian_result.mat', 'W', 'reg_loss', 'corr_XY', 'optimalParas')
