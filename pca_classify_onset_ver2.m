
lt = squeeze(mean(mean(left_trace,1),2));
rt = squeeze(mean(mean(right_trace,1),2));


%prepare the data by considering a span of 1 second (15 frames)  --> put in
%a variable norm_all 

rowtracker = 1;

for a = 0:15:4500
    disp(a);
    if a >= 4500
        break;
    end
    
    if a == 0
        rt1 = rt(a+1: a+15);
        lt1 = lt(a+1: a+15); 
        norm_all(rowtracker,1:15) = lt1;
        norm_all(rowtracker,16:30) = rt1;
    else 
        rt1 = rt(a:a+15-1);
        lt1 = lt(a:a+15-1);
        norm_all(rowtracker,1:15) = lt1;
        norm_all(rowtracker,16:30) = rt1;
    end
    rowtracker = rowtracker + 1; 
    %disp(rowtracker);
    
   
end

%prepare the label 

size_new = size(lt); 
labels_onset_old = zeros(size_new); 


for i = 1:29 
    labels_onset_old(onsets_right(i),1) = 1;

end  

for k = 1:23 
    labels_onset_old(onsets_left(k),1) = 1; 
    
end

rowtracker2 = 1 

for b = 0:15:4500
    if b >= 4500
        break;
    end
    
    if b == 0
        dummy = labels_onset_old(b+1:b+15,:);
    else 
        dummy = labels_onset_old(b:15+b,:);
    end
    
    if sum(dummy) == 0
        labels_onset_new(rowtracker2,:) = 0 
    else
        labels_onset_new(rowtracker2,:) = 1
    end
    rowtracker2 = rowtracker2 + 1 ; 
    
end
        
%prepare the data to fit in the classifier function input 



full_data = norm_all;
full_labels = labels_onset_new;
training_data = full_data(1:150,:);
testing_data = full_data(151:end,:);
training_labels = labels_onset_new(1:150,:);
testing_labels = labels_onset_new(151:end,:);

%running the data on SVM and KNN 
[kloss , percent] = svm(full_data,full_labels,training_data,testing_data,training_labels,testing_labels);

[kloss2 , percent2]= knn(6,full_data,full_labels,training_data,testing_data,training_labels,testing_labels);

disp("SVM");
disp(kloss);
disp(percent);

disp("KNN");
disp(kloss2);
disp(percent2);


%running data with pca + plotting 

[coeff,score,latent,~,explained] = pca(norm_all);

percents_SVM = [];
percent_k_SVM = [];
percents_KNN = [];
percent_k_KNN = [];



for x = 1:30

    full_data = score(:,1:x);
    full_labels = labels_onset_new;
    pca_data = score(:,1:x);
    training_data = pca_data(1:150,:)
    testing_data = pca_data(151:end,:);
    training_labels = labels_onset_new(1:150,:);
    testing_labels = labels_onset_new(151:end,:);
    
    
    [SVMkloss, percent_correct_SVM] = svm(full_data,full_labels,training_data,testing_data,training_labels,testing_labels);
    [KNNkloss,KNN_accuracy] = knn(5,full_data,full_labels,training_data,testing_data,training_labels,testing_labels);

    percents_SVM = [percents_SVM percent_correct_SVM];
    percent_k_SVM = [percent_k_SVM SVMkloss];
    percents_KNN = [percents_KNN KNN_accuracy];
    percent_k_KNN = [percent_k_KNN KNNkloss];

end 

%Plot SVM accuracy percent 

xvals_1 = 1:30;

figure

plot(xvals_1, percents_SVM)

title("SVM Classification Accuracy vs. Number of Components (Two Sided)")

xlabel("Number of Components From Each Side")

ylabel("Classifcation Accuracy")

%Plot SVM Kloss 

xvals_2 = 1:30;

figure

plot(xvals_2, percent_k_SVM)

title("SVM Classification KfoldLoss vs. Number of Components (Two Sided)")

xlabel("Number of Components From Each Side")

ylabel("K Fold Loss")


%Plot knn accuracy 

xvals_3 = 1:30;

figure

plot(xvals_3, percents_KNN)

title("KNN Classification Accuracy vs. Number of Components (Two Sided)")

xlabel("Number of Components From Each Side")

ylabel("Classifcation Accuracy")

%Plot knn K fold losses 

xvals_4 = 1:30;

figure

plot(xvals_4, percent_k_KNN)

title("KNN Classification kFoldLoss vs. Number of Components (Two Sided)")

xlabel("Number of Components From Each Side")

ylabel("K Fold Loss")


%%SVM AND KNN FUNCTION

function [kloss,accuracy_percent] = svm(full_data,full_labels,training_data,testing_data,training_labels,testing_labels)

%calculate kloss SVM
SVMModel_kloss = fitcsvm(full_data,full_labels);
CVSVMModel = crossval(SVMModel_kloss);
kloss = kfoldLoss(CVSVMModel);

%Calculate accuracy for predicting testing data using training data

SVMModel = fitcsvm(training_data,training_labels);

[LabelPredict,Score] = predict(SVMModel,testing_data);

%Check the differences between the output from predict (LabelPredict) and the actual
%TestingLabels

result = LabelPredict == testing_labels;

[a,b] = size(LabelPredict);

%calculate the accuracy of the classification

accuracy_percent = sum(result(:) == 1)/a * 100 ;

end


function [kloss,accuracy_percent] = knn(num_neighbour,full_data,full_labels,training_data,testing_data,training_labels,testing_labels)

% calculate kloss SVM
KNNModel_kloss = fitcknn(full_data,full_labels);
CVKNNModel = crossval(KNNModel_kloss);
kloss = kfoldLoss(CVKNNModel);

% Calculate accuracy for predicting testing data using training data

KNNModel = fitcknn(training_data,training_labels,'NumNeighbors',num_neighbour,'Standardize',1)


[Label,Score] = predict(KNNModel,testing_data)

%check the similarity of prediction from predict and the real outcome
%from testing data

result = Label == testing_labels;

[a,b] = size(Label);
number_of_1 = sum(result(:) == 1);

%calculate the accuracy KNN

accuracy_percent = number_of_1 /a * 100 ;

end






