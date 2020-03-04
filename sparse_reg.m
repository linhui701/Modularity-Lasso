function [w, objFun] = sparse_reg(X, Y, B, para, flag_abs, flag_penalty)
% ------------------------------------------------------------------------%
% Sparse Regression algorithm
% ------------------------------------------------------------------------%
% Input:
%       - X, Concatenated matrix contain proteinomics, genomics, SNPs
%       information.
%       - Y, pheno matrix
%       - Modularity Matrix: B
%       - group parameters, alpha, gamma(unknown, get from other functions)
%       - flag_abs: absoulte constraint for weights
%       - flag_penalty: 0: modularity 1: laplacian 2: lasso 3: elastic net
%                       4: lassoglm lasso
% Output:
%       - w, weight of X
%       - corrs, all corr of every iteration
%       - W, all w of every iteration
% ------------------------------------------------------------------------%
% Author: Linhui Xie, linhxie@iu.edu
% Date created: May-10-2019
% Update: Feb-22-2020
% @Indiana University School of Medicine.
% @Purdue University Electrical and Computer Engineering.
% ------------------------------------------------------------------------%


% passing parameters
alpha = para.alpha;
gamma = para.gamma;
n_concat_feature = size(X,2);
n_pheno = size(Y,2);

% ------------------------------------------------------------------------%
% Calculate coverance within concatenated feture data
XX = X'*X;
switch flag_penalty
    case 0      % 0: modularity
        Q = 2*(XX-alpha*B);
    case 1      % 1: laplacian
        Q = 2*(XX+alpha*B);
    case 2      % 2: lasso
        Q = 2*(XX);
end

% ------------------------------------------------------------------------%
% Calculate coverance between concatenated feture data and Pheno
XY = X'*Y(:,1);
C  = 2 * X' * Y(:,1);

%-------------------------------------------------------------------------%
% initialization
w = ones(n_concat_feature, 1) ./n_concat_feature; % initialize w here
objFun = [];
max_iter = para.max_iter; % pre-set, default set to 50
err = 1e-7; % 0.01 ~ 0.05
diff_obj = 10;

%-------------------------------------------------------------------------%
% analytical solution
analy_sol = Q\C;

%-------------------------------------------------------------------------%
% iterative solution
iter = 0; % counter
w_new = zeros(n_concat_feature,1);

while ((iter < max_iter) && ( err < diff_obj)) % default 50 times of iteration
    iter = iter+1;
    %---------------------------------------------------------------------%
    % soft-threshoding on w after 
    for j = 1:length(w)
        w_new(j)=sign(C(j)/Q(j,j)-sum( Q(j,:)/Q(j,j)*w-Q(j,j)*w(j) ) )*...
          max(0, abs(C(j)/Q(j,j)-sum( Q(j,:)/Q(j,j)*w-Q(j,j)*w(j) ) )- gamma/Q(j,j) );
    end
    
    %---------------------------------------------------------------------%
    % every iter, normalize the w to make |w|=1
    scale = sqrt(w_new'*w_new);
    w_new = w_new./scale; 
    
    if sum(isnan(w_new))
        w = w+eps;
        continue;
    end
    
    w = w_new;
    w_norm_one = sum(abs(w));
    
    %---------------------------------------------------------------------%
    % save results for every iteration
    switch flag_penalty 
        case 0      % 0: modularity
            constraint = - alpha*( w'*B*w );
        case 1      % 1: laplacian
            constraint =   alpha*( w'*B*w );
        case 2      % 2: lasso
            constraint =   0;
    end
    objFun(iter) = (Y-X*w)'*(Y-X*w) + gamma*w_norm_one + constraint;
    
    if iter ~= 1
        % relative prediction error
        diff_obj = abs((objFun(iter)-objFun(iter-1))/objFun(iter-1)); 
    end
end

