function  []=Q4(data,Xtrain,Xtest)


    %----------------train-------------------
    ytrain = data.ytrain;
    ytest = data.ytest;
    traindata = [Xtrain ytrain];
    [train_num,Yt] = size(traindata);
    [test_num,Yv] = size(Xtest);
    
    K=[1:10,15:5:100];
    % -----------------------test---------------------
   
            %-------test on train samples---------
            
   %put each test data in the whole train data to caculation the Euclidean distance
    dis_train=sqrt(Xtrain.^2*ones(Yv,train_num)+ones(train_num,Yv)*(Xtrain.^2)'-2*Xtrain*Xtrain');
    lengthK=length(K);

    nearpoints=zeros(K(lengthK),train_num,lengthK);
    
    for i=1:lengthK
        ytrain_expand=ytrain*ones(1,train_num);
        [M,I]=mink(dis_train',K(i));  %to find the nearest k points with different k values to the features that need to predict
        nearpoints(1:K(i),:,i)=ytrain_expand(I);
    end

    %caculation of the proportion of the spam class in K samples 
    predict_result=zeros(train_num,lengthK);

    for i=1:lengthK
        [row,col,v]=find(nearpoints(:,:,i)==1);

        for n=1:train_num
            count=length(find(col==n));
            if count/K(i)>=0.5            %if the proportion larger than or equal to 0.5, judge it as spam class(value is 1), else it is not spam class
                predict_result(n,i)=1;
            else
            end
        end
    end

    %compare the different predictive results with the test group, 

    errorcount=zeros(train_num,lengthK); 
    correct=zeros(1,lengthK);
    errorrate=zeros(1,lengthK);

    for u=1:lengthK
        errorcount(:,u)=xor(predict_result(:,u),ytrain);
        correct(u)=length(find(errorcount(:,u)==0));
        Error_train(u)=1-correct(u)/train_num;           %caculate the rate of error
    end
    
    
      % -----------------------test---------------------
   
            %-------test on test samples---------

%put each test data in the whole train data to caculation the Euclidean distance
    dis_test=sqrt(Xtest.^2*ones(Yv,train_num)+ones(test_num,Yv)*(Xtrain.^2)'-2*Xtest*Xtrain');

    nearpoints=zeros(K(lengthK),test_num,lengthK);

    %find the nearest k points with different k values to the features that need to predict ]

    for i=1:lengthK
        ytrain_expand=ytrain*ones(1,test_num);
        [M,I]=mink(dis_test',K(i));   %finds the indices of the k points and returns them in I.
        nearpoints(1:K(i),:,i)=ytrain_expand(I);
    end

    %caculation of the proportion of the spam class in K samples 
    predict_result=zeros(test_num,lengthK);

    for i=1:lengthK
        [row,col,v]=find(nearpoints(:,:,i)==1);

        for n=1:test_num
            count=length(find(col==n));
            if count/K(i)>=0.5          %if the proportion larger than or equal to 0.5, judge it as spam class(value is 1), else it is not spam class
                predict_result(n,i)=1;
            else
            end
        end
    end

    %compare the different predictive results with the test group and caculate the rate of error 
    errorcount=zeros(test_num,lengthK);
    correct=zeros(1,lengthK);
    Error_test=zeros(1,lengthK);

    for u=1:lengthK
        errorcount(:,u)=xor(predict_result(:,u),ytest);
        correct(u)=length(find(errorcount(:,u)==0));
        Error_test(u)=1-correct(u)/test_num;
    end

    % to Plot curves of training and test error rates versus K
    plot(K,Error_train*100,'r--','linewidth',1)
    hold on 
    plot(K,Error_test*100,'b-','linewidth',1)
    hold off
    grid on
    xlabel('K')
    ylabel('Error rate(%)')
    legend('training set','testing set','location','northwest')
    disp('K =1'),disp('train error rate'),disp(Error_train(1)),disp('test error rate'),disp(Error_test(1))
    disp('K =10'),disp('train error rate'),disp(Error_train(10)),disp('test error rate'),disp(Error_test(10))
    disp('K =100'),disp('train error rate'),disp(Error_train(end)),disp('test error rate'),disp(Error_test(end))
end




   