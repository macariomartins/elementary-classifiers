classdef HoldOut_Validation
    % HoldOut_Validation
    %
    % This is an implementation of Hold Out validation algorithm
    %
    properties
        trials = 1;                      % Number of trials to run with Hold Out
        Accuracies;                      % Vector of accuracies of each runned trial
        test_partition_percentage = 0.3; % The percentage of the 'Base' that must be splitted to use for testing purposes
    end
    
    methods
        function HoldOut = HoldOut_Validation(trials, test_partition_percentage)
            % HoldOut = HoldOut_Validation(trials, test_partition_percentage)
            %
            % This is the class construction function.
            %
            % @param trials is the number of times LOO must be runned
            % @param test_partition_percentage is the percentage of the 'Base' that must be splitted to use for testing purposes
            % @return HoldOut is a formatted object that will be returned
            %
            
            if (trials > 0)
                HoldOut.trials = trials;
            else
                fprintf("You need at least 1 trial. Using 1 trial now.");
            end
            if (test_partition_percentage > 0)
                HoldOut.test_partition_percentage = test_partition_percentage;
            else
                frpintf("Your test partition percentage must be major than zero! Using 0.30 percentage, instead.");
            end
            
            HoldOut.Accuracies = zeros(HoldOut.trials, 1);
        end
        
        function mean_accuracy = calculateAccuracy(HoldOut, Classifier, Base, Classes)
            % mean_accuracy = calculateAccuracy(HoldOut, Classifier, Base, Classes)
            %
            % This function calculates the mean accuracy for the Hold Out
            % trials. If only one trial is set, the accuracy returned will
            % be the obtained accuracy of the single running of validation.
            %
            % @param Classifier is a classifier already formatted object
            % @param Base is a set of attribute vectors to train the classifier
            % @param Classes is a set of classes assigned to each attribute vector in 'Base'. It is used to train the classifier, as well
            % @return mean_accuracy is the mean accuracy for the Hold Out trials
            %
            
            HoldOut.Accuracies = zeros(HoldOut.trials, 1);
            
            for i = 1:HoldOut.trials
                test_samples_number = size(Base, 1) * HoldOut.test_partition_percentage;
                [TestSamples, TestClasses, TrainingSamples, TrainingClasses] = HoldOut.splitBase(Base, Classes);
                
                for j = 1:test_samples_number
                    [class_marker, class_index] = max(TestClasses(j, :));
                    
                    test_sample = TestSamples(j, :);
                    test_class  = class_index;
                    
                    given_class = Classifier.classify(test_sample, TrainingSamples, TrainingClasses);
                    
                    if (given_class == test_class)
                        HoldOut.Accuracies(i) = HoldOut.Accuracies(i) + 1;
                    end
                end
                
                HoldOut.Accuracies(i) = HoldOut.Accuracies(i)/size(TestClasses, 1);
            end
            
            mean_accuracy = sum(HoldOut.Accuracies)/HoldOut.trials;
        end
        
        function [TestSamples, TestClasses, TrainingSamples, TrainingClasses] = splitBase(HoldOut, Base, Classes)
            % [TestSamples, TestClasses, TrainingSamples, TrainingClasses] = splitBase(HoldOut, Base, Classes)
            % 
            % This method split 'Base' and 'Classes', separating a selected
            % percentage to test purposes.
            %
            % @param Base is a set of attribute vectors
            % @param Classes is a set of classes assigned to each attribute vector in 'Base'
            % @return TestSamples is a set of attribute vectors for tests
            % @return TestClasses is a set of assigned classes to each attribute vector in 'TestSamples'
            % @return TrainingSamples is a set of attribute vectors for training
            % @return TrainingClasses is a set of assigned classes to each attribute vector in 'TrainingSamples'
            %
            
            test_samples_number = size(Base, 1) * HoldOut.test_partition_percentage;
            test_indexes = randi([1 size(Base, 1)], test_samples_number, 1);
            
            TestSamples = Base(test_indexes, :);
            TestClasses = Classes(test_indexes, :);
            
            Base(test_indexes, :) = [];
            Classes(test_indexes, :) = [];
            
            TrainingSamples = Base;
            TrainingClasses = Classes;
        end
    end
end
