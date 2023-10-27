function []    = Q2(data,Xtrain,Xtest)

ytrain=data.ytrain;
ytest=data.ytest;
traindata = [Xtrain ytrain];

%----------------train-------------------
    [train_num ,Yt] = size(traindata);
    [test_num ,Yv] = size(ytest);
    lambda = sum(ytrain)./train_num   ;    %lambda estimated by ML

    index_y0 = find(ytrain == 0);      %find the train samples given y = 0
    index_y1 = find(ytrain == 1);      %find the train samples given y = 1

    Xtrain_y1 = Xtrain(index_y1,:);          %train samples given y = 1
    Xtrain_y0 = Xtrain(index_y0,:);          %train samples given y = 0
    [num_y1 , ~] = size(Xtrain_y1);
    [num_y0 , numfea] = size(Xtrain_y0);
    u_theta_y1 = zeros(2,57) ;                  %store the class conditional(y=1) mean and variance of each feature  
    u_theta_y0 = zeros(2,57) ;                  %store the class conditional(y=0) mean and variance of each feature  
    u_theta_y1(1,:) = mean(Xtrain_y1);          % mean of feature when y = 1
    u_theta_y1(2,:) = var(Xtrain_y1)*(num_y1-1)/num_y1;      % variance of feature when y = 1
    u_theta_y0(1,:) = mean(Xtrain_y0);     % mean of feature when y = 0
    u_theta_y0(2,:) = var(Xtrain_y0)*(num_y0-1)/num_y0;      % variance of feature when y = 0





    %----------------------test---------------------
        %-------------test on train samples--------
    class_pre = zeros(train_num,4);
    class_pre(:,4) = ytrain;
    error_train = 0;
    for i = 1:train_num
        prob_y1 = 0; %The probility of y = 1
        prob_y0 = 0; %The probility of y = 0
        for j = 1:numfea
            prob_y1 = prob_y1 + log(normpdf(Xtrain(i,j),u_theta_y1(1,j),sqrt(u_theta_y1(2,j))));%The probility of y = 1
            prob_y0 = prob_y0 + log(normpdf(Xtrain(i,j),u_theta_y0(1,j),sqrt(u_theta_y0(2,j))));%The probility of y = 0
        end
        prob_y1 = prob_y1 + log(lambda); %The probility of y = 1
        prob_y0 = prob_y0 + log(1-lambda);%The probility of y = 0
        class_pre(i,1) = prob_y1;
        class_pre(i,2) = prob_y0;
       
        if prob_y1>prob_y0
            class_pre(i,3) = 1;       % prediction process
        end
        
        if class_pre(i,3) ~= class_pre(i,4) % to compare predict result with ture label
            error_train = error_train + 1;
        end
    end
    Error_train = error_train/train_num;



        %-------------test on test samples--------
    class_pre = zeros(test_num,4);
    class_pre(:,4) = ytest;
    error_test = 0;
    
    for i = 1:test_num;
        prob_y1 = 0; %The probility of y = 1
        prob_y0 = 0; %The probility of y = 0
        for j = 1:numfea;
            prob_y1 = prob_y1 + log(normpdf(Xtest(i,j),u_theta_y1(1,j),sqrt(u_theta_y1(2,j))));%The probility of y = 1
            prob_y0 = prob_y0 + log(normpdf(Xtest(i,j),u_theta_y0(1,j),sqrt(u_theta_y0(2,j))));%The probility of y = 0
        end
        prob_y1 = prob_y1 + log(lambda);%The probility of y = 1
        prob_y0 = prob_y0 + log(1-lambda);%The probility of y = 0
        class_pre(i,1) = prob_y1;
        class_pre(i,2) = prob_y0;
        
        %predict y with a higher probility
        if prob_y1>prob_y0
            class_pre(i,3) = 1;
        end
        
        %compare the predict result with true label
        if class_pre(i,3) ~= class_pre(i,4)
            error_test = error_test + 1;
        end
    end
    
    Error_test = error_test/test_num;
    disp('Error on training samples is:'),disp(Error_train)
    disp('Error on testing samples is:'),disp(Error_test)
end
