% Alex Samardzich <asamardz@stanford.edu>
% CS229a Final Project

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Load Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%{ 
Load the eligible and unclassified players files.  Note, the unclassified
players matrix contains all players who retired within the past 5 season,
but will be eligible for the Hall of Fame once enough time as passed since
their retirement.  It DOES NOT include players who have not played 5 full
seasons during their careers and are therefore ineligible for the HOF.
Load the X matrix which contains the shuffled eligible players matrix.
%}
load('players.mat','eligiblePlayers');
load('players.mat','unclassifiedPlayers');
load('players.mat','X');

tempEligiblePlayers = removevars(eligiblePlayers, {'HOF'});
playerIDs = [tempEligiblePlayers(:,1:5); unclassifiedPlayers(:,1:5)];

% Convert the tables into Matrices
eligiblePlayersMatrix = table2array(eligiblePlayers(:,4:end));
unclassifiedPlayersMatrix = table2array(unclassifiedPlayers(:,4:end));
unclassified_ID = table2array(unclassifiedPlayers(:,1));

%{
Shuffle the eligible players matrix rows so that there is no pattern
in the data before splitting it into the train, test, and validation sets.
This was done once and then saved into the players.mat so that each time
the file runs, the data isn't reshuffled which would lead to different
training set errors each time.

%EPs = [eligiblePlayersIDs, eligiblePlayersMatrix];
%X = EPs(randperm(size(EPs,1)),:);
%}

% Parameter to change whether or not downsizing takes place
downsizing = true;

% Downsizing non-HOF class 
if (downsizing)
    % The X matrix is sorted by HOF status, all HOF players are at the bottom
    % starting at index 5681
    X_temp = sortrows(X,2);
    X_non_HOF = X_temp(1:5680,:);
    X_HOF = X_temp(5681:end,:);

    % Shuffle the non_HOFers and then take part of the start of the 
    % matrix in order to downsize
    rng(1);
    X_non_HOF = X_non_HOF(randperm(size(X_non_HOF,1)),:);

    % Calculated how many non-HOFers to take
    desired_Class_Balance = 0.25;
    non_HOF_to_Take = round(size(X_HOF,1)/desired_Class_Balance);
    X_non_HOF = X_non_HOF(1:non_HOF_to_Take,:);

    % Combine the matrices and reshuffle
    X = [X_non_HOF; X_HOF];
    rng(1);
    X = X(randperm(size(X,1)),:);
end


% Extract the player IDs, the HOF Labels, and X
X_ID = X(:,1);
y = X(:,2);
X = X(:,3:end);

% Split into the different test sets
training_split = 0.7;
validation_split = 0.15;
testing_split = 1 - training_split - validation_split;

total_examples = size(y,1);
training_size = round(total_examples * training_split);
validation_size = round(total_examples * (training_split + validation_split));

X_ID_train = X_ID(1:training_size, :);
y_train = y(1:training_size, :);
X_train = X(1:training_size, :);

X_ID_validation = X_ID(training_size+1:validation_size , :);
y_validation = y(training_size+1:validation_size , :);
X_validation = X(training_size+1:validation_size , :);

X_ID_test = X_ID(validation_size+1:end, :);
y_test = y(validation_size+1:end, :);
X_test = X(validation_size+1:end, :);

% Normalize the datasets before training

[X_train_norm, mu, sigma] = featureNormalize(X_train);
X_validation_norm = (X_validation - mu)./sigma;
X_test_norm = (X_test - mu)./sigma;
unclassifiedPlayersMatrix_norm = (unclassifiedPlayersMatrix - mu)./sigma;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%% Logistic Regression %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set up the data matrix, add intercept term, and initialize parameters
[m, n] = size(X_train_norm);                      
X_train_norm_LR = [ones(m, 1) X_train_norm];
initial_theta = zeros(n + 1, 1); 
                      
X_validation_norm_LR = [ones(size(X_validation_norm, 1), 1) X_validation_norm];
                  
X_test_norm_LR = [ones(size(X_test_norm,1), 1) X_test_norm];

% Set options for fminunc                     
options = optimset('GradObj', 'on', 'MaxIter', 1000); 

% Select the parameter depending on whether or not downsizing is being used
if (downsizing)
    lambda_LR = 3.5;
else
    lambda_LR = 2;
end
     
% Run fminunc to obtain the optimal theta with or without regularization
%[theta, cost] = fminunc(@(t)(costFunction(t, X_train_norm, y_train)), initial_theta, options);
[theta, cost_LR] = fminunc(@(t)(costFunctionReg(t, X_train_norm_LR, y_train, lambda_LR)), initial_theta, options);

%{
% To find the optimal value of Lambda for regularization, a number of
% different values were tested and their corresponding Accuracy, Recall,
% Precision, and F1 scores were recorded for both the train and validation
% data sets.  These values where then compared to find the value of lambda
% that produced the best results.


lambdaVector1 = [0.001; 0.003; 0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30; 100; 300];

% Test the first lambdaVector to try to zero in on an approximate first
% value of lambda
%[lambdaTable1] = testLambdas(lambdaVector1, X_train_norm_LR, y_train,...
%     initial_theta, options, 0.5, X_validation_norm_LR, y_validation)

% The highest value for validation F1 occurs between lambdas of 1 and 3
% for no downsizing and between 3 and 10 with downsizing

%lambdaVector2 = [0.75; 1; 1.25; 1.5; 1.75; 2; 2.25; 2.5; 2.75; 3; 3.25];
lambdaVector2 = [3; 3.5; 4; 4.5; 5; 5.5; 6; 6.5; 7; 7.5; 8; 8.5; 9; 9.5; 10];

[lambdaTable2] = testLambdas(lambdaVector2, X_train_norm_LR, y_train,...
     initial_theta, options, 0.5, X_validation_norm_LR, y_validation)

% The second pass didn't seem to make a big difference in validation F1
% score, so a value of lambda = 2 was chosen for no downsizing.  With 
% downsizing, a value of 3.5 was selected
%}

threshold = 0.5;

%{
Potential thresholds were also tested to see if that parameter made a big
difference in success of the model.  After trying a number of values, it
was found the optimal threshold was 0.5.

thresholdVector = [0.3; 0.35; 0.4; 0.45; 0.5; 0.55; 0.6; 0.65; 0.7];

[thresholdTable] = testThresholds(thresholdVector, X_train_norm,...
    y_train, theta, X_validation_norm, y_validation)

%}


% Check theta on training set and calculate accuracy
p_train_LR = getPredictionVector(theta, X_train_norm_LR, threshold);                       
[Accuracy_train_LR, Precision_train_LR, Recall_train_LR, F1_train_LR] = ...
    performance(p_train_LR, y_train);
[False_Positives_Train_LR, False_Negatives_Train_LR]...
    = missclassifications(p_train_LR, y_train, X_ID_train, playerIDs);


% Check theta on validation set and calcualte accuracy
p_validation_LR = getPredictionVector(theta, X_validation_norm_LR, threshold);                       
[Accuracy_validation_LR, Precision_validation_LR, Recall_validation_LR, ...
    F1_validation_LR] = performance(p_validation_LR, y_validation);
[False_Positives_validation_LR, False_Negatives_validation_LR] = ...
    missclassifications(p_validation_LR, y_validation, X_ID_validation, playerIDs);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SVM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Train the SVM model using fitcsvm with a linear and gaussian kernel

%{
% Other tested kernels below.  They were not as promising as the linear or
% Gaussian kernel
%65% F1 on validation 
%SVM_Model = fitcsvm(X_train_norm, y_train);

%48% F1 on validation
%SVM_Model = fitcsvm(X_train_norm, y_train, 'KernelFunction', 'polynomial',...
%    'PolynomialOrder', 3);

% BoxConstraint, 0.012156 and KernelScale 0.5004 or 0.014235 and 0.51871
%SVM_Model = fitcsvm(X_train_norm, y_train,'OptimizeHyperparameters','auto',...
%    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%    'expected-improvement-plus'));

%Validation F1 68.85%
%SVM_Model = fitcsvm(X_train_norm, y_train, 'BoxConstraint', 0.012156,...
%    'KernelScale', 0.5004);

%}

%{
% Test different box constraint and kernel scale hyper parameters
%BoxConstraintVector = [0.001; 0.003; 0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30]; 
%KernelScaleVector = [0.1; 0.3; 1; 3; 10; 30; 100; 300; 1000; 3000];

% The first round of testing showed the optimal BoxConstraint was around
% 0.1 and the optimal KernelScale was around 1 with an F1 of 73.02
% Another peak occured at BC 1 and KS 3
BoxConstraintVector = [0.01; 0.05; 0.1; 0.5; 1; 1.5; 2; 2.5; 3]; 
KernelScaleVector = [0.5; 1; 1.5; 2; 2.5; 3];

% Second round suggested BC between 0.5 and 1 and KS between 2.5 and 3
%BoxConstraintVector = [0.5; 0.6; 0.7; 0.8; 0.9; 1]; 
%KernelScaleVector = [2.5; 2.6; 2.7; 2.8; 2.9; 3];

% Ideal pair was BC of 1 and KS of 2.5 with no downsizing. With downsizing
% BC is 0.05 and KS is 1
[trainF1Table, valF1Table] = testSVMParameters(BoxConstraintVector,...
KernelScaleVector, X_train_norm, y_train, X_validation_norm, y_validation, 0)
%}

if (downsizing)
    BC_Linear = 0.05;
    KS_Linear = 1;
else
    BC_Linear = 1;
    KS_Linear = 2.5;
end

% SVM with Linear Kernel
SVM_Model_Linear = fitcsvm(X_train_norm, y_train, 'BoxConstraint',...
    BC_Linear, 'KernelScale', KS_Linear);

[p_train_SVM_Linear, p_train_scores_SVM_Linear] = predict(SVM_Model_Linear, X_train_norm);
[Accuracy_train_SVM_Linear, Precision_train_SVM_Linear, Recall_train_SVM_Linear, F1_train_SVM_Linear]...
    = performance(p_train_SVM_Linear, y_train);
[False_Positives_train_SVM_Linear, False_Negatives_train_SVM_Linear] = ...
    missclassifications(p_train_SVM_Linear, y_train, X_ID_train, playerIDs);

[p_validation_SVM_Linear, p_validation_scores_SVM_Linear] = predict(SVM_Model_Linear, X_validation_norm);
[Accuracy_validation_SVM_Linear, Precision_validation_SVM_Linear, Recall_validation_SVM_Linear, F1_validation_SVM_Linear]...
    = performance(p_validation_SVM_Linear, y_validation);
[False_Positives_validation_SVM_Linear, False_Negatives_validation_SVM_Linear] = ...
    missclassifications(p_validation_SVM_Linear, y_validation, X_ID_validation, playerIDs);

% Retry with a Gaussian Kernel
%{
% Tune the BoxConstraint and KernelScale
%BoxConstraintVector = [10; 30; 100; 300; 1000; 3000; 10000; 30000]; 
%KernelScaleVector = [10; 30; 100; 300; 1000; 3000; 10000; 30000];

% First round peaked at BC 3000 and KS 300 without downsizing.  With
% downsizing the peak is BC 1500 and KS 100
BoxConstraintVector = [1000; 1500; 2000; 2500; 3000; 3500; 4000]; 
KernelScaleVector = [50; 100; 150; 200; 250; 300; 350; 400];
[trainF1Table, valF1Table] = testSVMParameters(BoxConstraintVector,...
KernelScaleVector, X_train_norm, y_train, X_validation_norm, y_validation, 1)
%}

if (downsizing)
    BC_Gaussian = 1500;
    KS_Gaussian = 100;
else
    BC_Gaussian = 3000;
    KS_Gaussian = 300;
end

SVM_Model_Gaussian = fitcsvm(X_train_norm, y_train, 'KernelFunction', 'gaussian',...
    'BoxConstraint', BC_Gaussian, 'KernelScale', KS_Gaussian);

[p_train_SVM_Gaussian, p_train_scores_SVM_Gaussian] = predict(SVM_Model_Gaussian, X_train_norm);
[Accuracy_train_SVM_Gaussian, Precision_train_SVM_Gaussian, Recall_train_SVM_Gaussian, F1_train_SVM_Gaussian]...
    = performance(p_train_SVM_Gaussian, y_train);
[False_Positives_train_SVM_Gaussian, False_Negatives_train_SVM_Gaussian] = ...
    missclassifications(p_train_SVM_Gaussian, y_train, X_ID_train, playerIDs);

[p_validation_SVM_Gaussian, p_validation_scores_SVM_Gaussian] = predict(SVM_Model_Gaussian, X_validation_norm);
[Accuracy_validation_SVM_Gaussian, Precision_validation_SVM_Gaussian, Recall_validation_SVM_Gaussian, F1_validation_SVM_Gaussian]...
    = performance(p_validation_SVM_Gaussian, y_validation);
[False_Positives_validation_SVM_Gaussian, False_Negatives_validation_SVM_Gaussian] = ...
    missclassifications(p_validation_SVM_Gaussian, y_validation, X_ID_validation, playerIDs);



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Neural Network %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set input layer, hidden layer, and output layer sizes
input_layer_size  = size(X_train_norm,2);  
hidden_layer_size = 50;   
output_layer_size = 1;

% Initialize weights
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, output_layer_size);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% Set parameters for fmin
options = optimset('MaxIter', 600);
if (downsizing)
    lambda_NN = 2.5;
else
    lambda_NN = 3;
end

%{
% Test various values of lambda to see which gives the best validation
% set F1 score
% Peaks for validation F1 score at 0.1 and 3
%lambdaVector2 = [0.05; 0.1; 0.15; 0.2; 0.25; 1.5; 2; 2.5; 3; 3.5; 4; 4.5; 5];
lambdaVector2 = [1; 1.5; 2; 2.5; 3; 3.5; 4; 4.5; 5; 5.5; 6; 6.5];

% Peak at lambda = 0.25 and 3, 3 is probably best for the test set since it
% is higher regularization so less variance
[lambdaTable] = testLambdasNN(lambdaVector2, X_train_norm, y_train,...
    input_layer_size, hidden_layer_size, output_layer_size,...
    options, 0.5, X_validation_norm, y_validation, initial_nn_params)
%}
                                                   
costFunctionNN = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
                                   output_layer_size, X_train_norm, y_train, lambda_NN);
                               
[nn_params, cost_NN] = fmincg(costFunctionNN, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 output_layer_size, (hidden_layer_size + 1));
             
cost_NN(size(cost_NN,1));
             
% Check performance of NN
p_train_NN = predictNN(Theta1, Theta2, X_train_norm, 0.5);
[Accuracy_train_NN, Precision_train_NN, Recall_train_NN, F1_train_NN]...
    = performance(p_train_NN, y_train);
[False_Positives_train_NN, False_Negatives_train_NN] = ...
    missclassifications(p_train_NN, y_train, X_ID_train, playerIDs);

p_validation_NN = predictNN(Theta1, Theta2, X_validation_norm, 0.5);
[Accuracy_validation_NN, Precision_validation_NN, Recall_validation_NN, F1_validation_NN]...
    = performance(p_validation_NN, y_validation);
[False_Positives_validation_NN, False_Negatives_validation_NN] = ...
    missclassifications(p_validation_NN, y_validation, X_ID_validation, playerIDs);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Logistic Regression Test Set
p_test_LR = getPredictionVector(theta, X_test_norm_LR, threshold);                       
[Accuracy_test_LR, Precision_test_LR, Recall_test_LR, ...
    F1_test_LR] = performance(p_test_LR, y_test);
[False_Positives_test_LR, False_Negatives_test_LR] = ...
    missclassifications(p_test_LR, y_test, X_ID_test, playerIDs);

% Linear SVM Test Set
[p_test_SVM_Linear, p_test_scores_SVM_Linear] = predict(SVM_Model_Linear, X_test_norm);
[Accuracy_test_SVM_Linear, Precision_test_SVM_Linear, Recall_test_SVM_Linear, F1_test_SVM_Linear]...
    = performance(p_test_SVM_Linear, y_test);
[False_Positives_test_SVM_Linear, False_Negatives_test_SVM_Linear] = ...
    missclassifications(p_test_SVM_Linear, y_test, X_ID_test, playerIDs);

% Gaussian SVM Test Set
[p_test_SVM_Gaussian, p_test_scores_SVM_Gaussian] = predict(SVM_Model_Gaussian, X_test_norm);
[Accuracy_test_SVM_Gaussian, Precision_test_SVM_Gaussian, Recall_test_SVM_Gaussian, F1_test_SVM_Gaussian]...
    = performance(p_test_SVM_Gaussian, y_test);
[False_Positives_test_SVM_Gaussian, False_Negatives_test_SVM_Gaussian] = ...
    missclassifications(p_test_SVM_Gaussian, y_test, X_ID_test, playerIDs);

% Neural Network Test Set
p_test_NN = predictNN(Theta1, Theta2, X_test_norm, 0.5);
[Accuracy_test_NN, Precision_test_NN, Recall_test_NN, F1_test_NN]...
    = performance(p_test_NN, y_test);
[False_Positives_test_NN, False_Negatives_test_NN] = ...
    missclassifications(p_test_NN, y_test, X_ID_test, playerIDs);

% Predict current players who are likely going to make the Hall of Fame
% using the highest test set model, the SVM with a Linear Kernel
[p_unclassified_players, p_unclassified_players_scores] = predict(SVM_Model_Linear, unclassifiedPlayersMatrix_norm);
[Future_HOF] = FutureHOF(p_unclassified_players, unclassified_ID, playerIDs)

% Plot the F1 Scores for the various models

F1_Scores = [Precision_test_LR, Recall_test_LR, F1_test_LR;...
    Precision_test_SVM_Linear, Recall_test_SVM_Linear F1_test_SVM_Linear;...
    Precision_test_SVM_Gaussian, Recall_test_SVM_Gaussian, F1_test_SVM_Gaussian;...
    Precision_test_NN, Recall_test_NN, F1_test_NN];

model_Names = categorical({'Logistic Reg.','SVM Linear','SVM Gaussian','Neural Network'});
model_Names = reordercats(model_Names,...
    {'Logistic Reg.','SVM Linear','SVM Gaussian','Neural Network'});

figure(1)
bar(model_Names, F1_Scores)
set(gca,'fontsize',10);
%title('Test Set Performance by Model')
ylabel('F1 Score');
ylim([0.85 1])
xlabel('Models');
legend({'Precision','Recall', 'F1'});
legend('Location','northeast');
set(gca,'fontname','Palatino');
set(gcf,'color','w');
xtickangle(0);

% Confusion Matrix for Gaussian SVM
figure(2)
C = confusionmat(y_test, p_test_SVM_Linear);
groups = {'Non-HOF','HOF'};
cm = confusionchart(C, groups);
%cm.Title = 'SVM Linear - Test Set Confusion Matrix';
set(gca,'FontSize',14);
set(gca,'FontName','Palatino');
set(gcf,'color','w');

%{
% Find features with the highest values in the theta matrix 
max_theta = maxk(theta,5);
theta_indices = zeros(size(max_theta));
for i = 1:size(theta_indices)
   theta_indices(i) = find(theta==max_theta(i));
end

for j = 1:size(theta_indices)
    eligiblePlayers(1,theta_indices(j)+5)
end

% Most predictive stats are: 1. Approximate Value, 2. Number of Punts,
% 3. Pro Bowl Selections, 4. Passing Attempts, 5. Passes Defended
%}


%%
%%%%%%%%%%%%%%%%%%%%%%   Functions Used in This File   %%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%  Sigmoid Function  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Calculates the sigmoid activiation of input variable z
function g = sigmoid(z)
    
g = 1 ./ (1 + exp(-z));

end

%%%%%%%%%%%%%%%%%%%%%%%%%  Normalize Features  %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Takes input matrix X and returns a normalized version of X along with
% the mean and standard deviations used to normalize the matrix
function [X_norm, mu, sigma] = featureNormalize(X)

sigma = std(X);
mu = mean(X);
X_norm = (X - mu)./sigma;

end

%%%%%%%%%%%%%%%%%%%%%%%%%  LR Cost Function  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% CostFunction for logistic regression
function [J, grad] = costFunction(theta, X, y)

m = size(y,1);                   % number of training examples
J = 0;                           % set it correctly
grad = zeros(size(theta));       % set it correctly

J = (1 / m) * (-y'*log(sigmoid(X*theta))-(1-y)'*log(1-sigmoid(X*theta)));
grad = (1 / m) * X' * (sigmoid(X*theta)-y);

end

%%%%%%%%%%%%%%%%  LR Cost Function W/ Regularization  %%%%%%%%%%%%%%%%%%%%%

% Cost Function with Regularization
function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y);                % Number of training examples
J = 0;                        % Set J to the regularized cost in logistic regression
grad = zeros(size(theta));    % Set grad to the regularized gradient of logistic regression

temp_theta = theta;
temp_theta(1) = 0;
J = (1 / m) * (-y'*log(sigmoid(X*theta))-(1-y)'*log(1-sigmoid(X*theta))) + (lambda / (2 * m)) * (temp_theta' * temp_theta);
grad = (1 / m) * X' * (sigmoid(X*theta)-y) + (lambda / m) * temp_theta;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%  LR Prediction  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Predict to get HOF chances for each player in X
function p = getPredictionVector(theta, X, threshold)

m = size(X, 1);                                         
p = sigmoid(X * theta) >= threshold;

end

%%%%%%%%%%%%%%%%%%%%%%%%%  Check Performance  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute Accuracy, Recall, Precision, and F1 Score
function [Accuracy, Precision, Recall, F1] = performance(p, y)

check = p - y;
falsePositives = sum(check == 1);
falseNegatives = sum(check == -1);
truePositives = sum(p) - falsePositives;

Accuracy = mean(double(p == y)) * 100;
Precision = truePositives / (truePositives + falsePositives);
Recall = truePositives / (truePositives + falseNegatives);
F1 = 2 * Precision * Recall / (Precision + Recall);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%  Error Analysis  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Print the name of false positives and false negatives
function [False_Positives, False_Negatives] = missclassifications(p, y, X_ID, playerID)

check = y - p;
falseNeg = find(check == 1);
falsePos = find(check == -1);

if (size(falsePos,1) == 0)
    False_Positives = "No False Positives";
else    
    False_Positives = playerID(X_ID(falsePos(1)),:); 
end

if (size(falsePos,1) > 1)
    for i=2:size(falsePos,1)
        False_Positives = [False_Positives; playerID(X_ID(falsePos(i)),:)];
    end   
    False_Positives = sortrows(False_Positives, 'From_3');
end


if (size(falseNeg,1) == 0)
    False_Negatives = "No False Negatives";
else    
    False_Negatives = playerID(X_ID(falseNeg(1)),:); 
end

if (size(falseNeg,1) > 1)
    for i=2:size(falseNeg,1)
        False_Negatives = [False_Negatives; playerID(X_ID(falseNeg(i)),:)];
    end   
    False_Negatives = sortrows(False_Negatives, 'From_3');
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%  Future HOFers  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Print the name of positive examples among unclassified players
function [Future_HOF] = FutureHOF(p, X_ID, playerID)

indices = find(p == 1);
num_indices = size(indices,1);

if (num_indices < 1)
    return
end

% plus 1 to indices becuase there is a gap between the eligible players
% and the unclassified players in the matrix
Future_HOF = playerID(X_ID(indices(1)+1),:);

if (num_indices < 2)
    return
end

for i=2:num_indices
        Future_HOF = [Future_HOF; playerID(X_ID(indices(i)+1),:)];
end

Future_HOF = sortrows(Future_HOF, 'From_3');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%  LR Test Lambdas  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function used to test various lambda values in logistic regression
function [lambdaTable] = testLambdas(lambdaVector, X_train_norm, y_train,...
    initial_theta, options, threshold, X_validation_norm, y_validation)

trainAccuracy = zeros(size(lambdaVector,1),1);
trainPrecision = zeros(size(lambdaVector,1),1);
trainRecall = zeros(size(lambdaVector,1),1);
trainF1 = zeros(size(lambdaVector,1),1);
trainCost = zeros(size(lambdaVector,1),1);
validationAccuracy = zeros(size(lambdaVector,1),1);
validationPrecision = zeros(size(lambdaVector,1),1);
validationRecall = zeros(size(lambdaVector,1),1);
validationF1 = zeros(size(lambdaVector,1),1);

for i=1:size(lambdaVector,1)
    
    
[theta, cost] = fminunc(@(t)(costFunctionReg(t, X_train_norm, y_train,...
    lambdaVector(i))), initial_theta, options);
    
p_train = getPredictionVector(theta, X_train_norm, threshold);                       
[Accuracy_train, Precision_train, Recall_train, F1_train]...
    = performance(p_train, y_train);

trainAccuracy(i) = Accuracy_train;
trainPrecision(i) = Precision_train;
trainRecall(i) = Recall_train;
trainF1(i) = F1_train;
trainCost(i) = cost;

p_validation = getPredictionVector(theta, X_validation_norm, threshold);                       
[Accuracy_validation, Precision_validation, Recall_validation, ...
    F1_validation] = performance(p_validation, y_validation);

validationAccuracy(i) = Accuracy_validation;
validationPrecision(i) = Precision_validation;
validationRecall(i) = Recall_validation;
validationF1(i) = F1_validation;
    
end


lambdaTable = table(lambdaVector, trainAccuracy, trainPrecision,...
    trainRecall, trainF1, trainCost, validationAccuracy,...
    validationPrecision, validationRecall, validationF1);

end

%%%%%%%%%%%%%%%%%%%%%%%%  LR Test Thresholds  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function used to test tresholds for predicting positive
function [thresholdTable] = testThresholds(thresholdVector, X_train_norm,...
    y_train, theta, X_validation_norm, y_validation)

trainAccuracy = zeros(size(thresholdVector,1),1);
trainPrecision = zeros(size(thresholdVector,1),1);
trainRecall = zeros(size(thresholdVector,1),1);
trainF1 = zeros(size(thresholdVector,1),1);
validationAccuracy = zeros(size(thresholdVector,1),1);
validationPrecision = zeros(size(thresholdVector,1),1);
validationRecall = zeros(size(thresholdVector,1),1);
validationF1 = zeros(size(thresholdVector,1),1);

for i=1:size(thresholdVector,1)

p_train = getPredictionVector(theta, X_train_norm, thresholdVector(i));                       
[Accuracy_train, Precision_train, Recall_train, F1_train]...
    = performance(p_train, y_train);

trainAccuracy(i) = Accuracy_train;
trainPrecision(i) = Precision_train;
trainRecall(i) = Recall_train;
trainF1(i) = F1_train;

p_validation = getPredictionVector(theta, X_validation_norm, thresholdVector(i));                       
[Accuracy_validation, Precision_validation, Recall_validation, ...
    F1_validation] = performance(p_validation, y_validation);

validationAccuracy(i) = Accuracy_validation;
validationPrecision(i) = Precision_validation;
validationRecall(i) = Recall_validation;
validationF1(i) = F1_validation;
    
end


thresholdTable = table(thresholdVector, trainAccuracy, trainPrecision,...
    trainRecall, trainF1, validationAccuracy, validationPrecision,...
    validationRecall, validationF1);

end

%%%%%%%%%%%%%%%%%%%%%%  Test BC and KS for SVM  %%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function used to test various SVM parameters in order to determine the
% optimal BoxConstraint and KernelScale
function [trainF1Table, valF1Table] = testSVMParameters(BoxConstraintVector,...
    KernelScaleVector, X_train_norm, y_train, X_validation_norm, y_validation, gaussian)

trainF1Table = zeros(size(BoxConstraintVector,1),size(KernelScaleVector,1));
valF1Table = zeros(size(BoxConstraintVector,1),size(KernelScaleVector,1));

for i=1:size(BoxConstraintVector,1)
    for j=1:size(KernelScaleVector,1)
        tic
        if (gaussian ==1)
            model = fitcsvm(X_train_norm, y_train,'KernelFunction', 'gaussian',...
            'BoxConstraint', BoxConstraintVector(i), 'KernelScale',...
            KernelScaleVector(j));
        else
            model = fitcsvm(X_train_norm, y_train, 'BoxConstraint',...
            BoxConstraintVector(i), 'KernelScale', KernelScaleVector(j));
        end

        [p_train_SVM, ~] = predict(model, X_train_norm);
        [~, ~, ~, F1_train_SVM] = performance(p_train_SVM, y_train);
        trainF1Table(i,j) = F1_train_SVM;
        
        [p_validation_SVM, ~] = predict(model, X_validation_norm);
        [~, ~, ~, F1_validation_SVM] = performance(p_validation_SVM, y_validation);
        valF1Table(i,j) = F1_validation_SVM;
        
        toc
        BoxConstraintVector(i)
        KernelScaleVector(j)
    end
end

end

%%%%%%%%%%%%%%%%%%%%  Neural Network Cost Function  %%%%%%%%%%%%%%%%%%%%%%%

% Cost function with regularization for a neural network with 1 hidden layer
function [J, grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                                   output_layer_size, X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 output_layer_size, (hidden_layer_size + 1));

m = size(X, 1);                                                                       

% PART 1: FEED FORWARD PROPAGATION

a_1 = [ones(size(X,1),1) X]; 
z_2 = a_1*Theta1';
a_2 = sigmoid(z_2);
a_2 = [ones(size(a_2, 1),1) a_2]; 
z_3 = a_2*Theta2';
a_3 = sigmoid(z_3);
J = (1 / m) * (-y'*log(a_3) - (1-y)'*log(1-a_3)); 
%J = sum(sum(J));
%grad = (1 / m) * X' * (sigmoid(X*theta)-y);

temp_T1 = (Theta1.^2);
temp_T1(:,1) = 0;
temp_T2 = (Theta2.^2);
temp_T2(:,1) = 0; 


reg = (lambda / (2 * m)) * (sum(sum(temp_T1)) + sum(sum(temp_T2)));

J = J + reg;

% PART 2: BACK PROPAGATION

d_3 = a_3 - y;
d_2 = d_3*Theta2.*(a_2.*(1-a_2));
d_2 = d_2(:,2:end);

Theta1_grad = (1/m) * d_2' * a_1;
Theta2_grad = (1/m) * d_3' * a_2;

tempTheta1 = Theta1;
tempTheta1(:,1) = 0;
tempTheta2 = Theta2;
tempTheta2(:,1) = 0;

Theta1_grad = Theta1_grad + (lambda/m) * tempTheta1;
Theta2_grad = Theta2_grad + (lambda/m) * tempTheta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];     

end


%%%%%%%%%%%%%%%%%%%%%  Randomly Initialize Weights  %%%%%%%%%%%%%%%%%%%%%%%

% Randomly itilizaes weights for use in a neural network
function W = randInitializeWeights(L_in, L_out)                               

% Randomly initialize the weights to small values
% Use the same random kernel so that it's possible to compare performances
% across multiple runnings of the program to tune parameters
rng(1);
epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;

end

%%%%%%%%%%%%%%%%%%%%%  Predict for a Neural Network  %%%%%%%%%%%%%%%%%%%%%%

% Returns a prediction vector for the neural network
function p = predictNN(Theta1, Theta2, X, threshold)
m = size(X, 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
p = sigmoid([ones(m, 1) h1] * Theta2');
p = p >= threshold;

end

%%%%%%%%%%%%%%%%%%%  Test Lambdas for Neural Network  %%%%%%%%%%%%%%%%%%%%%

% Test various values of lambda for use in a neural network
function [lambdaTable] = testLambdasNN(lambdaVector, X_train_norm, y_train,...
    input_layer_size, hidden_layer_size, output_layer_size,...
    options, threshold, X_validation_norm, y_validation, initial_nn_params)

trainAccuracy = zeros(size(lambdaVector,1),1);
trainPrecision = zeros(size(lambdaVector,1),1);
trainRecall = zeros(size(lambdaVector,1),1);
trainF1 = zeros(size(lambdaVector,1),1);
trainCost = zeros(size(lambdaVector,1),1);
validationAccuracy = zeros(size(lambdaVector,1),1);
validationPrecision = zeros(size(lambdaVector,1),1);
validationRecall = zeros(size(lambdaVector,1),1);
validationF1 = zeros(size(lambdaVector,1),1);

for i=1:size(lambdaVector,1)
  costFunctionNN = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
                                   output_layer_size, X_train_norm, y_train, lambdaVector(i));
                               
[nn_params, cost_NN] = fmincg(costFunctionNN, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 output_layer_size, (hidden_layer_size + 1));
             
p_train_NN = predictNN(Theta1, Theta2, X_train_norm, threshold);
[Accuracy_train, Precision_train, Recall_train, F1_train]...
    = performance(p_train_NN, y_train);

trainAccuracy(i) = Accuracy_train;
trainPrecision(i) = Precision_train;
trainRecall(i) = Recall_train;
trainF1(i) = F1_train;
trainCost(i) = cost_NN(size(cost_NN,1));

p_validation = predictNN(Theta1, Theta2, X_validation_norm, threshold);                       
[Accuracy_validation, Precision_validation, Recall_validation, ...
    F1_validation] = performance(p_validation, y_validation);

validationAccuracy(i) = Accuracy_validation;
validationPrecision(i) = Precision_validation;
validationRecall(i) = Recall_validation;
validationF1(i) = F1_validation;
    
end

lambdaTable = table(lambdaVector, trainAccuracy, trainPrecision,...
    trainRecall, trainF1, trainCost, validationAccuracy,...
    validationPrecision, validationRecall, validationF1);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  fmincg  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function used to minimize cost for the neural network
function [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
% Minimize a continuous differentialble multivariate function. Starting point
% is given by "X" (D by 1), and the function named in the string "f", must
% return a function value and a vector of partial derivatives. The Polack-
% Ribiere flavour of conjugate gradients is used to compute search directions,
% and a line search using quadratic and cubic polynomial approximations and the
% Wolfe-Powell stopping criteria is used together with the slope ratio method
% for guessing initial step sizes. Additionally a bunch of checks are made to
% make sure that exploration is taking place and that extrapolation will not
% be unboundedly large. The "length" gives the length of the run: if it is
% positive, it gives the maximum number of line searches, if negative its
% absolute gives the maximum allowed number of function evaluations. You can
% (optionally) give "length" a second component, which will indicate the
% reduction in function value to be expected in the first line-search (defaults
% to 1.0). The function returns when either its length is up, or if no further
% progress can be made (ie, we are at a minimum, or so close that due to
% numerical problems, we cannot get any closer). If the function terminates
% within a few iterations, it could be an indication that the function value
% and derivatives are not consistent (ie, there may be a bug in the
% implementation of your "f" function). The function returns the found
% solution "X", a vector of function values "fX" indicating the progress made
% and "i" the number of iterations (line searches or function evaluations,
% depending on the sign of "length") used.
%
% Usage: [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
%
% See also: checkgrad 
%
% Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
%
%
% (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
% 
% Permission is granted for anyone to copy, use, or modify these
% programs and accompanying documents for purposes of research or
% education, provided this copyright notice is retained, and note is
% made of any changes that have been made.
% 
% These programs and documents are distributed without any warranty,
% express or implied.  As the programs were written for research
% purposes only, they have not been tested to the degree that would be
% advisable in any important application.  All use of these programs is
% entirely at the user's own risk.
%
% [ml-class] Changes Made:
% 1) Function name and argument specifications
% 2) Output display
%

% Read options
if exist('options', 'var') && ~isempty(options) && isfield(options, 'MaxIter')
    length = options.MaxIter;
else
    length = 100;
end


RHO = 0.01;                            % a bunch of constants for line searches
SIG = 0.5;       % RHO and SIG are the constants in the Wolfe-Powell conditions
INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
EXT = 3.0;                    % extrapolate maximum 3 times the current bracket
MAX = 20;                         % max 20 function evaluations per line search
RATIO = 100;                                      % maximum allowed slope ratio

argstr = ['feval(f, X'];                      % compose string used to call function
for i = 1:(nargin - 3)
  argstr = [argstr, ',P', int2str(i)];
end
argstr = [argstr, ')'];

if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
S=['Iteration '];

i = 0;                                            % zero the run length counter
ls_failed = 0;                             % no previous line search has failed
fX = [];
[f1 df1] = eval(argstr);                      % get function value and gradient
i = i + (length<0);                                            % count epochs?!
s = -df1;                                        % search direction is steepest
d1 = -s'*s;                                                 % this is the slope
z1 = red/(1-d1);                                  % initial step is red/(|s|+1)

while i < abs(length)                                      % while not finished
  i = i + (length>0);                                      % count iterations?!

  X0 = X; f0 = f1; df0 = df1;                   % make a copy of current values
  X = X + z1*s;                                             % begin line search
  [f2 df2] = eval(argstr);
  i = i + (length<0);                                          % count epochs?!
  d2 = df2'*s;
  f3 = f1; d3 = d1; z3 = -z1;             % initialize point 3 equal to point 1
  if length>0, M = MAX; else M = min(MAX, -length-i); end
  success = 0; limit = -1;                     % initialize quanteties
  while 1
    while ((f2 > f1+z1*RHO*d1) || (d2 > -SIG*d1)) && (M > 0) 
      limit = z1;                                         % tighten the bracket
      if f2 > f1
        z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3);                 % quadratic fit
      else
        A = 6*(f2-f3)/z3+3*(d2+d3);                                 % cubic fit
        B = 3*(f3-f2)-z3*(d3+2*d2);
        z2 = (sqrt(B*B-A*d2*z3*z3)-B)/A;       % numerical error possible - ok!
      end
      if isnan(z2) || isinf(z2)
        z2 = z3/2;                  % if we had a numerical problem then bisect
      end
      z2 = max(min(z2, INT*z3),(1-INT)*z3);  % don't accept too close to limits
      z1 = z1 + z2;                                           % update the step
      X = X + z2*s;
      [f2 df2] = eval(argstr);
      M = M - 1; i = i + (length<0);                           % count epochs?!
      d2 = df2'*s;
      z3 = z3-z2;                    % z3 is now relative to the location of z2
    end
    if f2 > f1+z1*RHO*d1 || d2 > -SIG*d1
      break;                                                % this is a failure
    elseif d2 > SIG*d1
      success = 1; break;                                             % success
    elseif M == 0
      break;                                                          % failure
    end
    A = 6*(f2-f3)/z3+3*(d2+d3);                      % make cubic extrapolation
    B = 3*(f3-f2)-z3*(d3+2*d2);
    z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3));        % num. error possible - ok!
    if ~isreal(z2) || isnan(z2) || isinf(z2) || z2 < 0 % num prob or wrong sign?
      if limit < -0.5                               % if we have no upper limit
        z2 = z1 * (EXT-1);                 % the extrapolate the maximum amount
      else
        z2 = (limit-z1)/2;                                   % otherwise bisect
      end
    elseif (limit > -0.5) && (z2+z1 > limit)         % extraplation beyond max?
      z2 = (limit-z1)/2;                                               % bisect
    elseif (limit < -0.5) && (z2+z1 > z1*EXT)       % extrapolation beyond limit
      z2 = z1*(EXT-1.0);                           % set to extrapolation limit
    elseif z2 < -z3*INT
      z2 = -z3*INT;
    elseif (limit > -0.5) && (z2 < (limit-z1)*(1.0-INT))  % too close to limit?
      z2 = (limit-z1)*(1.0-INT);
    end
    f3 = f2; d3 = d2; z3 = -z2;                  % set point 3 equal to point 2
    z1 = z1 + z2; X = X + z2*s;                      % update current estimates
    [f2 df2] = eval(argstr);
    M = M - 1; i = i + (length<0);                             % count epochs?!
    d2 = df2'*s;
  end                                                      % end of line search

  if success                                         % if line search succeeded
    f1 = f2; fX = [fX' f1]';
    %fprintf('%s %4i | Cost: %4.6e\r', S, i, f1);
    s = (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;      % Polack-Ribiere direction
    tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
    d2 = df1'*s;
    if d2 > 0                                      % new slope must be negative
      s = -df1;                              % otherwise use steepest direction
      d2 = -s'*s;    
    end
    z1 = z1 * min(RATIO, d1/(d2-realmin));          % slope ratio but max RATIO
    d1 = d2;
    ls_failed = 0;                              % this line search did not fail
  else
    X = X0; f1 = f0; df1 = df0;  % restore point from before failed line search
    if ls_failed || i > abs(length)          % line search failed twice in a row
      break;                             % or we ran out of time, so we give up
    end
    tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
    s = -df1;                                                    % try steepest
    d1 = -s'*s;
    z1 = 1/(1-d1);                     
    ls_failed = 1;                                    % this line search failed
  end
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
end
end