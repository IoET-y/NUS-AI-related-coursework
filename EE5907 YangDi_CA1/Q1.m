function []    = Q1(data,Xtrain,Xtest)
ytrain=data.ytrain;
ytest=data.ytest;
traindata = [Xtrain ytrain];

%----------------------------train---------------------------------
    [train_num ,Yt] = size(traindata);        %train_num: the number of train samples
    lambda = sum(ytrain)./train_num   ;    %lambda estimated by ML
    P_train = zeros(4,57);           % given y, various feature x and its elements in trainning set

    index_y0 = find(ytrain == 0);      %index of the features given y = 0
    index_y1 = find(ytrain == 1);      %index the features given y = 1
    [train_numy1,b] = size(index_y1);      %total train number of y = 1      
    [train_numy0,b] = size(index_y0);      %total train number of y = 0
    
    for i = 1:57;
        feature = Xtrain(:,i);           %A single feature Xi of all train samples
        feature_x1y1 = feature(index_y1);    %these train samples of which Xi is 1 given y = 1
        feature_x1y0 = feature(index_y0);    %these train samples of which Xi is 1 given y = 0
        num_x1y1 = sum(feature_x1y1);  %when y = 1 , the number of train samples whose Xi = 1
        num_x1y0 = sum(feature_x1y0);  %when y = 0 , the number of train samples whose Xi = 1                            
       % train_numy1 - num_x1y1;  when y = 1 , the number of train samples whose Xi = 0
       % train_numy0 - num_x1y0:  when y = 0 , the number of train samples whose Xi = 0
       
        P_train(:,i)=[num_x1y1; train_numy1 - num_x1y1; num_x1y0; train_numy0 - num_x1y0];
    end




    %-------------------------test-------------------------


    Error_test = [] ;      %errorcount in test set
    Error_train = [] ;     %errorcount in train set

    for alpha = 0:0.5:100 ; %alpha varys from 0 to 100, step 0.5.
        [test_num,count0] = size(Xtest);
        
        
        %---------------test on test set-------------------

        
        P_test_y1 = zeros(test_num,3);   % probability of y = 1 in test set
        for j = 1:test_num                
            prob = 0;  % probability of y = 1 in test set
            sample = Xtest(j,:);
            for i = 1:57
                if sample(i) == 1
                    prob = prob + log((P_train(1,i)+ alpha)/(train_numy1 + 2*alpha));
                else
                    prob = prob + log((P_train(2,i)+ alpha)/(train_numy1 + 2*alpha));
                end
            end
            prob = prob + log(lambda);
            P_test_y1(j,1) = prob ;
        end  
     % probability of y = 0 in test set
        for j = 1:test_num                
            prob = 0;
            sample = Xtest(j,:);
            for i = 1:57
                if sample(i) == 1
                    prob = prob + log((P_train(3,i)+alpha)/(train_numy0 + 2*alpha));
                else
                    prob = prob + log((P_train(4,i)+alpha)/(train_numy0 + 2*alpha));
                end
            end
            prob = prob + log(1-lambda);
            P_test_y1(j,2) = prob ;
        end
    % predice y = 1 or y = 0 in test set
        for i = 1:test_num                  
            if P_test_y1(i,1) > P_test_y1(i,2)
                P_test_y1(i,3) = 1;
            else
                P_test_y1(i,3) = 0;
            end
        end
    % compute test error rate
        errorcount = 0;
        for i = 1:test_num        
            if ytest(i) ~= P_test_y1(i,3)
                errorcount = errorcount + 1;
            end
        end
        error_rate = errorcount / test_num;
        Error_test = [Error_test error_rate]; % to form a matrix contain error_rates corresponding to various alpha





        %----------------------test on train set---------------

        P_train_y1 = zeros(train_num,3); % predicted probability of y = 1 in train set

        for k = 1:train_num
            sample = Xtrain(k,:);
            prob = 0;
            for s = 1:57
                if sample(s) == 1
                    prob = prob + log((alpha + P_train(1,s))/(2*alpha + train_numy1));
                else
                    prob = prob + log((alpha + P_train(2,s))/(2*alpha + train_numy1));
                end
            end
            prob = prob + log(lambda);
            P_train_y1(k,1) = prob;
        end

        % predicted probability of y = 0 in train set
        for k = 1:train_num
            sample = Xtrain(k,:);
            prob = 0;
            for s = 1:57
                if sample(s) == 1
                    prob = prob + log((alpha + P_train(3,s))/(2*alpha + train_numy0));
                else
                    prob = prob + log((alpha + P_train(4,s))/(2*alpha + train_numy0));
                end
            end
            prob = prob + log(1-lambda);
            P_train_y1(k,2) = prob;
        end
        % predicted y = 1 or y = 0 in train set
        for i = 1:train_num                  
            if P_train_y1(i,1) > P_train_y1(i,2)
                P_train_y1(i,3) = 1;
            else
                P_train_y1(i,3) = 0;
            end
        end
        
      % compute train error rate
        errorcount = 0;
        for i = 1:train_num        
            if ytrain(i) ~= P_train_y1(i,3)
                errorcount = errorcount + 1;
            end
        end
        error_rate = errorcount / train_num;
        Error_train = [Error_train error_rate]; % to form a matrix contain error_rates corresponding to various alpha             

    end


    % Plots the curve of training and test error rates versus α
    alpha = [];
    for i = 0:0.5:100
        alpha = [alpha i];
    end    
    plot(alpha,Error_train*100,'r--','linewidth',1)
    hold on
    plot(alpha,Error_test*100,'b-','linewidth',1)
    hold off
    grid on
    xlabel('α')
    ylabel('Error rate(%)')
    legend('training set','testing set','location','northwest')
    title('Error rates VS various α')


    disp('alpha = 1:'),disp('train error rate'),disp(Error_train(3)),disp('test error rate'),disp(Error_test(3))
    disp('alpha = 10:'),disp('train error rate'),disp(Error_train(21)),disp('test error rate'),disp(Error_test(21))
    disp('alpha = 100:'),disp('train error rate'),disp(Error_train(201)),disp('test error rate'),disp(Error_test(201))
end
           
            
        














  