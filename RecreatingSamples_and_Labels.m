%Average the right and left trace and squeeze it to 2 dimension

lt = squeeze(mean(mean(left_trace,1),2));
rt = squeeze(mean(mean(right_trace,1),2));

%normalize for onset right
 for a = 1:29
     rt1 = rt(onsets_right(a):(onsets_right(a)+30-1));
     lt1 = lt(onsets_right(a):(onsets_right(a)+30-1));
     norm_rt1 = (rt1-mean(rt1))./std(rt1);
     norm_lt1 = (lt1-mean(lt1))./std(lt1);
     norm_onset_right(a,1:30) = transpose(norm_lt1)
     norm_onset_right(a,31:60)= transpose(norm_rt1)
     
     
 end
 
 % normalize onset left
  for b = 1:23
      rt2 = rt(onsets_left(b):(onsets_left(b)+30-1));
      lt2 = lt(onsets_left(b):(onsets_left(b)+30-1));
      norm_rt2  = (rt2-mean(rt2))./std(rt2);
      norm_lt2 = (lt2-mean(lt2))./std(lt2);
      norm_onset_left(b,1:30) = transpose(norm_lt2)
      norm_onset_left(b,31:60)= transpose(norm_rt2)
     
  end
  
%loop through onset right and left 
%onsets_right = 1, onsets_left = 0
% 
 i = 1;
 k = 1;
%  
rowtracker = 1;
 samples_recreate = zeros(52,60 );
 labels_recreate = zeros(52,1);
% 
 while ((i <= 29)&& (k <= 23))
    if onsets_right(i) < onsets_left(k)
        labels_recreate(rowtracker,1) = 1;
        samples_recreate(rowtracker,1:60) = norm_onset_right(i,:)
        rowtracker = rowtracker + 1;
        i = i + 1 ;
        
    else 
        labels_recreate(rowtracker,1) = 0;
        samples_recreate(rowtracker,1:60) = norm_onset_left(k,:)
        rowtracker = rowtracker + 1; 
        k = k+1;
        
    end

end        