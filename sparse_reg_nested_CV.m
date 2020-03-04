function [W, reg_loss, corr_XY, main_para] = sparse_reg_nested_CV(X, Y, B, ...
    para, n_folds, inner_folds, flag_abs, flag_penalty)
% ------------------------------------------------------------------------%
% Nested Cross Validation, tuning parameters
% ------------------------------------------------------------------------%
% Input:
%       - X, concatanated feature data for n subjects
%       - Y, cognitive test score for n subjects
%       - Modularity Matrix: B
%       - group parameters, alpha, gamma(unknown, get from other functions)
%       - folds, number of folds
%       - inner_folds, for nested cross validation
%       - flag_abs: absoulte constraint for weights
%       - flag_penalty: 0: modularity 1: laplacian 2: lasso 3: elastic net
%                       4: lassoglm lasso
% Output:
%       - W, weight of sparse linear regression
%       - corr_XY, correlation, w1'*X1'*X2*w2
%       - res, all results
%       - optimalParas, optimal parameters.
% Original Author: Jingwen Yan
% @Indiana University School of Medicine.
% Modified Author: Linhui Xie, linhxie@iu.edu, xie215@purdue.edu
% Last update: 05/19/2019 
% ------------------------------------------------------------------------%

disp('=====================================');
disp('Begin nested cross validition...');
disp('=====================================');
% ------------------------------------------------------------------------%
% Initilization
w = 0;
corr_XY = 0;
[n_sbj, n_concat_feature] = size(X);
n_cog_test = size(Y,2);
str_title={'B','L','Lasso','Elastic','Lasso2'};
abs_title={'without_abs','with_abs'};
reg_loss=zeros(3,5);
corr_XY=zeros(1,5);

% ------------------------------------------------------------------------%
% Generating folds

% Seed with the same seed every time while setting the outside folds.
% create the same fold to compare the result.
rng(10); % fix seed for reproducibility

if length(n_folds) == 1
    [indices, folds] = sparse_reg_fold_generation(n_sbj,n_folds);
end
folds_num = max(indices);

% ------------------------------------------------------------------------%
% set the loss at very huge level
opt_reg_loss = 10^7;
objFuc = zeros(folds_num, para.max_iter);
for k = 1:folds_num
    % use the fold corresponding to the current loop num
    % as final evaluation for the regression loss
    sprintf('%dth fold is running...',k)
    test = (indices == k); 
     
    train = ~test;
    
    X_train = X(train,:);
    Y_train = Y(train,:);
    
    X_test = X(test,:);
    Y_test = Y(test,:);

    if flag_penalty < 3
        % Tuning parameters
        t = cputime;
        if isempty(inner_folds)
            % first loop, there is no inner_folds, so run this command.
            [optimalParas,inner_folds] = sparse_reg_tune_parameter(X_train,...
                Y_train, B, para, folds_num, k, flag_abs, flag_penalty);
        else
            % for 2nd,3rd,4th loop, the inner_folds contain 209 subjects
            % X_train contain 209 subjects as well
            
            % for 5th loop, the inner_folds should contain 212 subjects, but
            % after 1st loop, the inner_folds contain 209 subjects incices
            % X_train contain 212 subjects as well
            [optimalParas,inner_folds] = sparse_reg_tune_parameter(X_train,...
                Y_train, B, para, inner_folds, k, flag_abs, flag_penalty);
        end
        
        disp(strcat('Tunning parameter takes time: ',num2str(cputime-t)));
        
        [w_train, objFuc(k,:)] = sparse_reg(X_train, Y_train, B, ...
            optimalParas, flag_abs, flag_penalty);
        
    elseif flag_penalty == 3
        % Elastic net result
        rng('default')  % for reproducibility
        
        alpha = 0.5;
        [W, FitInfo] = lassoglm( X_train, Y_train, 'normal', ...
            'CV', 5, 'Alpha', alpha);
        
        % step 3, find appropriate regularization
        lassoPlot(W, FitInfo, 'PlotType', 'CV');
        legend('show', 'Location', 'best');
        
        lambda = FitInfo.LambdaMinDeviance;
        index  = FitInfo.IndexMinDeviance;
        W0     = FitInfo.Intercept(index);
        
        w_train = W(:,index);
        optimalParas.lambda = lambda;
        optimalParas.alpha = alpha;
    elseif flag_penalty == 4
        % Another lasso result
        rng('default')  % for reproducibility
        
        [W, FitInfo] = lassoglm( X_train, Y_train, 'normal', 'CV', 5);
        
        % step 3, find appropriate regularization
        lassoPlot(W, FitInfo, 'PlotType', 'CV');
        legend('show', 'Location', 'best');
        
        lambda = FitInfo.LambdaMinDeviance;
        index  = FitInfo.IndexMinDeviance;
        W0     = FitInfo.Intercept(index);
        
        w_train = W(:,index);
        optimalParas.lambda = lambda;
    end
    
    % --------------------------------------------------------------------%
    Y_pred = X_test*w_train;
    % Mean-Squared-Root-Error
    msre_reg_loss = sqrt((Y_test-Y_pred)'*(Y_test-Y_pred)/length(Y_test));
    
    % Mean-Absolute-Error
    mae_reg_loss = sum(abs(Y_test-Y_pred)/length(Y_test));
    
    % R-Squared
    Rsq2 = 1 - sum((Y_test - Y_pred).^2)/sum((Y_test - mean(Y_test)).^2);

    figure('Visible','Off')
    plot(1:para.max_iter,objFuc(k,:))
    print('-dpng',sprintf('%d_%f_%s.png',k, msre_reg_loss, ...
        str_title{flag_penalty+1})) ;
    
    W(:,k) = w_train;
    reg_loss(1,k) = msre_reg_loss;
    reg_loss(2,k) = mae_reg_loss;
    reg_loss(3,k) = Rsq2;
    corr_XY(1,k) =  corr(Y_test, X_test*w_train );
    main_para(:,k) = optimalParas;
end

if flag_penalty < 3
    figure('Visible','Off')
    plot(1:para.max_iter,objFuc)
    print('-dpng',sprintf('all_five_fold_in_a_plot_%s.png',...
        str_title{flag_penalty+1}) );
end
