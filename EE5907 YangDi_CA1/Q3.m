function []    = Q3(data,Xtrain,Xtest)

    ytrain = data.ytrain;
    ytest = data.ytest;
    traindata = [Xtrain ytrain];
    %----------------------------train--------------------------------
    [train_num feature_num] = size(Xtrain);   %number of train samples , number of features
    [test_num feature_num] = size(Xtest);     %number of test samples , number of features
    bias_term = ones(train_num,1);      % bias term
    Xtrain_bias = [bias_term Xtrain];   %add bias term to original train dataset
    Xtest_bias = [bias_term(1:test_num) Xtest];  %add bias term to original test dataset
    T_Xtrain_bias = Xtrain_bias';

 %-----------------------Training processing----------------------------

    lambda = [1:9,10:5:100];

    H1 = eye(feature_num+1,feature_num+1);   %  The 2nd term of hessian matrix with regulatization
    H1(1,1) = 0;
    I = eye(feature_num+1);
    I(1,1) = 0;
    wlist = ones(feature_num+1,length(lambda));  % store w of corresponding lambda

    for i = 1:length(lambda)
        w = zeros(1,feature_num+1);
        w = w';
        iteration = 1;
        NLLlist = [];   %prepare to store NLL of corresponding lambda
        
%------------Given w and lamda，compute the NLL with regularization-----------
        while iteration == 1
            [train_num,fea_num] = size(Xtrain_bias);
            w_Tx = Xtrain_bias*w;
            NLL = 0;
            
            for j = 1:train_num %compute NLL
                NLL = NLL +ytrain(j) * log(1/(1+exp(-w_Tx(j)))) + (1-ytrain(j)) * log(1/(1+exp(w_Tx(j))));
            end
            
            w1 = w';
            w1 = w1(2:end); %Do not regularization on the bias term
            NLL = -NLL + 0.5*lambda(i) * w1 * w1'; %NLL with regularization               
            NLLlist = [NLLlist NLL];

            
 %---------- Given Xtrain_bias and w to compute the μ vector-------------- 
           [trainb_num row] = size(Xtrain_bias);
            w_Tx = Xtrain_bias * w;
            U = ones(trainb_num,1);
            for j = 1:trainb_num
                U(j) = 1/(1+exp(-w_Tx(j)));
            end
           
            W = w;
            W(1) = 0;
            grad = T_Xtrain_bias * (U-ytrain) + lambda(i) * W;   %gradient
            

%-----------Given thje vector μ,compute the corresponding diagonal matrix S
           
            S = zeros(length(U));
            for j = 1:length(U)
                S(j,j) = U(j)*(1-U(j));
            end
            
            H = T_Xtrain_bias * S * Xtrain_bias + lambda(i) * I;   %Hessian
            w = w - (inv(H) * grad);                   %update w
            
            if length(NLLlist)>1                %convergence condition
                if abs(NLLlist(end) - NLLlist(end-1)) < 0.01    %If the NLL reduces very slightly between 2 successive iteration
                    iteration = 0;                              %We think it converges
                end
            end
        end
        wlist(:,i) = w;
    end

    
    
    %-----------------------test-----------------------

    %---------------test on train samples----------------
    Error_train = ones(1,length(lambda));
    for i = 1:length(lambda)
        errorcount = 0;
        Xtrain_pred = Xtrain_bias * (-wlist(:,i));
        for j = 1:train_num
            if Xtrain_pred(j) <= 0  %   P(y=1) = 1/(1+exp(-wx)) > 0.5 equals that -wx ＜ 0
                Xtrain_pred(j) =1;
            else
                Xtrain_pred(j) = 0;
            end
            
            if Xtrain_pred(j) ~= ytrain(j)
                errorcount = errorcount + 1;
            end
        end
        Error_train(i) = errorcount/train_num;     %coXtest_prempare the different predictive results with the test group and caculate the rate of errorcount 
    end

      %---------------test on test samples---------------------
    Error_test = ones(1,length(lambda));
    for i = 1:length(lambda);
        errorcount = 0;
        Xtest_pred = Xtest_bias*(-wlist(:,i));
        for j = 1:test_num;
            if Xtest_pred(j) <= 0;
                Xtest_pred(j) =1;
            else
                Xtest_pred(j) = 0;
            end

            if Xtest_pred(j) ~= ytest(j)
                errorcount = errorcount + 1;
            end
        end
        Error_test(i) = errorcount/test_num;    % compare the different predictive results with the test set and caculate the rate of errorcount 
    end      


    plot(lambda,Error_train*100,'r--','linewidth',1)
    hold on
    plot(lambda,Error_test*100,'b-','linewidth',1)
    hold off
    grid on
    xlabel('λ')
    ylabel('Error rate(%)')
    legend('training set','testing set','location','northwest')
    title('Error rates VS λ')

    disp('lambda =1'),disp('train error rate'),disp(Error_train(1)),disp('test error rate'),disp(Error_test(1))
    disp('lambda =10'),disp('train error rate'),disp(Error_train(10)),disp('test error rate'),disp(Error_test(10))
    disp('lambda =100'),disp('train error rate'),disp(Error_train(end)),disp('test error rate'),disp(Error_test(end))
end

