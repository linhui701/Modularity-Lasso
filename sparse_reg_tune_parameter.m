function [optimalPara,indices_inner] = sparse_reg_tune_parameter(X, Y, B, para, folds_num, k, flag_abs, flag_penalty)
%-------------------------------------------------------------------------%
% Tuning parameters by grid search
%-------------------------------------------------------------------------%
% Input:
%       - X, Concatenated matrix contain proteinomics, genomics, SNPs
%       information.
%       - Y, pheno matrix
%       - Modularity Matrix: B
%       - group parameters, alpha, gamma(unknown, get from other functions)
%       - folds_num, number of folds
%       - k, k-th fold
%       - inner_folds, for nested cross validation
%       - flag_abs: absoulte constraint for weights
%       - flag_penalty: 0: modularity 1: laplacian 2: lasso 3: elastic net
%                       4: lassoglm lasso
% Output:
%       - w, weight of sparse regression model
%       - flag_abs: absoulte constraint for weights
%       - flag_penalty: 0: modularity 1: laplacian 2: lasso 3: elastic net
%                       4: lassoglm lasso
%-------------------------------------------------------------------------%

disp('=====================================');
disp('Begin tuning parameters...');
disp('=====================================');

%-------------------------------------------------------------------------%
% initialization
n_sbj = size(X,1);
w = 0;
corr_XY = 0;

%-------------------------------------------------------------------------%
% Set parameters

alpha = para.alpha;
gamma = para.gamma;

alpha_num = length(alpha);
gamma_num = length(gamma);

%-------------------------------------------------------------------------%
% Set the first candidate as default parameter
alpha_optimal = alpha(1);
gamma_optimal = gamma(1);

%-------------------------------------------------------------------------%
% Generating folds for inner loop of trainning data
if length(folds_num) == 1
    % for 1st loop, there is no inner_folds, hence need to generate inner
    % fold indices for 209 subjects, 5 new folds indices for 209 subjects 
    % are generated
    
    % random permutation for inner fold generation indices_inner(1)
    rng('shuffle')
    indices_inner = sparse_reg_fold_generation(n_sbj,folds_num);
    
elseif length(folds_num) == n_sbj
    % for 2nd,3rd,4th loop, the <folds_num=inner_folds=indices_inner(1)>  
    % contain 209 subjects fold indices, 
    % hence resign <indices_inner> with value of <folds_num>
    % X_train contain 209 subjects as well
    indices_inner = folds_num;
    
else
    % for 5th loop, <folds_num = inner_folds = indices_inner(1)>  
    % contain 209 subjects fold indices,
    % however, we have 212 subjects, hence need to regenerated inner fold
    folds_num = max(folds_num);
    indices_inner = sparse_reg_fold_generation(n_sbj,folds_num);
end

folds_num = max(indices_inner);

%-------------------------------------------------------------------------%
% Begin Loop
disp('----------------------------------------------------------');
disp('Begin searching the optimal parameters...');
disp('----------------------------------------------------------');
% set default objFuc as very big number
tmp_loss_optimal = 10^7;

for i = 1:alpha_num

    sprintf('%dth iteration for \x03b1',i); % unicode 03b1 is for alpha
    
    for j = 1:gamma_num
        
        reg.max_iter = para.max_iter;
        objFun =  [];
        reg.alpha = alpha(i);
        reg.gamma = gamma(j);
        % Inner loop - tuning parameters
        
        tmp_loss =  zeros(folds_num,1);
        for kk = 1:folds_num
            tmp_objFun = [];
            test_idx = (indices_inner == kk);
            train_idx = ~test_idx;
            
            X_train = X(train_idx,:);
            X_test  = X(test_idx,:);
            Y_train = Y(train_idx,:);
            Y_test  = Y(test_idx,:);
            
            % Inner - call gcca algorithm
            [w, tmp_objFun] = sparse_reg(X_train, Y_train, B, reg, flag_abs, flag_penalty);
            
            objFun(kk,1:length(tmp_objFun)) = tmp_objFun;
            
            Y_pred = X_test*w;
            tmp_loss(kk,1) = sqrt((Y_test-Y_pred)'*(Y_test-Y_pred)/length(Y_test));
        end
                
        if mean(tmp_loss) < tmp_loss_optimal
            alpha_optimal = reg.alpha;
            gamma_optimal = reg.gamma;
            tmp_loss_optimal= mean(tmp_loss);
        end
    end
end
disp('| Searching the optimal parameters done |');
disp('----------------------------------------------------------');

% Output the optimal parameters
optimalPara.alpha = alpha_optimal;
optimalPara.gamma = gamma_optimal;
optimalPara.max_iter = para.max_iter;


