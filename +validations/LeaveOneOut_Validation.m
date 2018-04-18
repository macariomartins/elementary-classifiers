classdef LeaveOneOut_Validation
    % LeaveOneOut_Validation
    %
    % This class is an implementation of the Leave-One-Out cross validation
    % method.
    %
    properties
        trials = 1; % Number of trials to run with LOO
        Accuracies; % Vector of accuracies of each runned trial
    end
    
    methods
        function LOO = LeaveOneOut_Validation(trials)
            % LOO = LeaveOneOut_Validation(trials)
            %
            % This is the class construction function.
            %
            % @param trials is the number of times LOO must be runned
            % @return LOO is a formatted object that will be returned
            %
            
            if (trials > 0)
                LOO.trials = trials;
            else
                fprintf("You need at least 1 trial. Using 1 trial now.");
            end
            
            LOO.Accuracies = zeros(LOO.trials, 1);
        end
        
        function mean_accuracy = calculateAccuracy(LOO, Classifier, Base, Classes)
            % mean_accuracy = calculateAccuracy(LOO, Classifier, Base, Classes)
            %
            % This function calculates the mean accuracy for the LOO trials
            % if only one trial is set, the accuracy returned will clearly
            % be the obtained accuracy of the single running of LOO cross
            % validation.
            %
            % @param Classifier is a classifier already formatted object
            % @param Base is a set of attribute vectors to train the classifier
            % @param Classes is a set of classes assigned to each attribute vector in 'Base'. It is used to train the classifier, as well
            % @return mean_accuracy is the mean accuracy for the LOO trials
            %
            
            LOO.Accuracies = zeros(LOO.trials, 1);
            
            for i = 1:LOO.trials
                for j = 1:size(Base, 1)
                    Base_cp = Base;
                    Classes_cp = Classes;
                    
                    [class_marker, class_index] = max(Classes_cp(j, :));
                    
                    test_sample = Base_cp(j, :);
                    test_class  = class_index;
                    
                    Base_cp(j, :) = [];
                    Classes_cp(j, :) = [];
                    
                    TrainingSamples = Base_cp;
                    TrainingClasses = Classes_cp;
                    
                    given_class = Classifier.classify(test_sample, TrainingSamples, TrainingClasses);
                    
                    if (given_class == test_class)
                        LOO.Accuracies(i) = LOO.Accuracies(i) + 1;
                    end
                end
                
                LOO.Accuracies(i) = LOO.Accuracies(i)/size(Base, 1);
            end
            
            mean_accuracy = sum(LOO.Accuracies)/LOO.trials;
        end 
    end
end
