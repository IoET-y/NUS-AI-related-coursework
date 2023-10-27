function [Xtrain,Xtest] = dataprocess(strategy,data,threshold)

OG_Xtrain = data.Xtrain;
OG_Xtest  = data.Xtest;

%strategy a: log-transform: transform each feature using log(xij + 0.1) (assume natural log)

       if strategy == 1 % 1 refering to strategy a 
          Xtrain = log(OG_Xtrain+0.1);  %training dataset after transformation
          Xtest = log(OG_Xtest+0.1);    %test dataset after transformation
          
 %strategy b: binarization: binarize features: I(xij > 0). right here, the threhold value is set to 0 by default. 
 %If a feature is greater than 0,it's simply set to 1. If it’s less than or equal to 0, it is set to 0.

  else if strategy == 2  % 2 refering to strategy b 

          [datanum,column] = size(OG_Xtrain); 
            for i = 1:datanum
                for j = 1:column
                    if OG_Xtrain(i,j) > threshold
                        OG_Xtrain(i,j) = 1;
                    end
                end
            end
            Xtrain = OG_Xtrain; %training dataset after transformation
            

         [datanum,column] = size(OG_Xtest); 
            for i = 1:datanum
                for j = 1:column
                    if OG_Xtest(i,j) > threshold
                        OG_Xtest(i,j) = 1;
                    end
                end
            end
          Xtest = OG_Xtest; %test dataset after transformation  
      else
          disp('useless strategy')
      end
end
