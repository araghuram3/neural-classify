function NeuroClassifyAssignment1
    % Part 1  
    % kNN model
    load('mouse_data');
    
    % labels 
    all_onsets = [onsets_left, onsets_right];
    
    % assigning onsets left a class of 0 and onsets right a class of 1
    unsorted_labels = [zeros(1,length(onsets_left)), ones(1,length(onsets_right))];
    
    % sorts the onsets in order but also has the indicies for labels
    [sorted_onsets, inds] = sort(all_onsets); 
    my_labels = transpose(unsorted_labels(inds));
    
    % samples
    my_samples = [];
    lt = squeeze(mean(mean(left_trace,1),2));
    rt = squeeze(mean(mean(right_trace,1),2));

    for i = 1:length(sorted_onsets)
        lt1 = lt(sorted_onsets(i):(sorted_onsets(i)+30-1));
        rt1 = rt(sorted_onsets(i):(sorted_onsets(i)+30-1));
        norm_lt1 = (lt1-mean(lt1))./std(lt1);
        norm_rt1 = (rt1-mean(rt1))./std(rt1);
        sample = transpose([norm_lt1;  norm_rt1]);
        my_samples = [my_samples; sample];
    end
    
    % svm for my_labels and my_samples
    full_predictive_model = fitcsvm(my_samples,my_labels);
    cross_validated_model = crossval(full_predictive_model);
    loss = kfoldLoss(cross_validated_model);
    disp(loss);
end 