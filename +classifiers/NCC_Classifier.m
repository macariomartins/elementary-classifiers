classdef NCC_Classifier
    % NCC_Classifier
    %
    % This class is an implementation of the Nearest Centroid Classifier
    %
    properties
        distance_method = "euclidean";     % The distance calculation method to use
        allowed_distances = ["euclidean"]; % Distance methods allowed implemented in this class
        use_zscore = true;                 % Whether to use or not the z-score normalization
        Distances;                         % Distances from the 'sample' to each centroid
    end
    
    methods
        function NCC = NCC_Classifier(distance_method, use_zscore)
            % NCC = NCC_Classifier(distance_method, use_zscore)
            %
            % This is the class construction function.
            %
            % @param distance_method is a string that must exists in allowed_distances
            % @param use_zscore is a boolean to choose whether zscore normalization should be used or not
            % @return NCC is a formatted object that will be returned
            %
            
            if (size(find(NCC.allowed_distances == distance_method, 1, 'last'), 2))
                NCC.distance_method = distance_method;
            else
                fprintf("The selected distance '%s' could not be used. Using 'euclidean' instead.", distance_method);
            end
            
            NCC.use_zscore = use_zscore;
        end
        
        function class = classify(NCC, sample, Base, Classes)
            % class = classify(NCC, sample, Base, Classes)
            %
            % The main function of a classifier. The method classify return a
            % class index in order to classify the sent sample.
            %
            % @param sample is an attribute vector to be classified
            % @param Base is a set of attribute vectors to train the classifier
            % @param Classes is a set of classes assigned to each attribute vector in 'Base'. It is used to train the classifier, as well
            % @return class a scalar class index assigned to 'sample' as a classification result
            %
            
            Centroids = zeros(size(Classes, 2), size(Base, 2));
            CentroidClasses = eye(size(Classes, 2));
            
            for i = 1:size(Classes, 2)
                class_indexes = find(Classes(:, i) == 1);
                ClassBase = Base(class_indexes, :);
                Centroids(i, :) = sum(ClassBase)/size(ClassBase, 1);
            end
            
            NCC.Distances = NCC.calculateDistances(sample, Centroids);
            nearest_centroid_index = NCC.findNearestCentroid(NCC.Distances);
            
            [value index] = max(Classes(nearest_centroid_index, :));
            class = index;
        end
        
        function centroid_index = findNearestCentroid(NCC, Distances)
            % centroid_index = findNearestCentroid(NCC, Distances)
            %
            % This method search for the nearest centroid to 'sample',
            % considering the given 'Distances'.
            %
            % @param Distances a vector that represents the distance of 'sample' to the classes centroids
            % @return centroid_index a scalar class index assigned to 'sample' by the nearest centroid
            %
            
            [value index]  = min(Distances);
            centroid_index = index;
        end
        
        function Distances = calculateDistances(NCC, sample, Centroids)
            % Distances = calculateDistances(NCC, sample, Centroids)
            %
            % This function calculate the distance from 'sample' to the classes
            % centroids using a given distance calculation method and applying
            % z-score to data, if defined by the user.
            %
            % @param sample is an attribute vector to be classified
            % @param Centroids is a set of attribute vectors centroids to train the classifier
            % @return Distances a vector that represents the distance of 'sample' to the classes centroids
            %
            
            if (NCC.use_zscore)
                X = [sample; Centroids];
                Z = NCC.zscoreIt(X);
                sample = Z(1, :);
                Centroids   = Z(2:size(Z, 1), :);
            end
            
            switch (NCC.distance_method)
                case "euclidean"
                    Distances = NCC.euclideanDistance(sample, Centroids);
                otherwise
                    Distances = NCC.euclideanDistance(sample, Centroids);
            end
        end
        
        function Z = zscoreIt(NCC, X)
            % Z = zscoreIt(NCC, X)
            %
            % Apply the z-score normalization to the matrix 'X'.
            %
            % @param X is a matrix where each column represents an attribute and each line represents a sample
            % @return Z a zscored matrix calculated from X
            %
            
            [rows cols] = size(X);
            Z = zeros(rows, cols);
            Means = sum(X)/rows;
            StdDeviations = std(X);
            
            for j = 1:cols
                for i = 1:rows
                    Z(i, j) = (X(i, j) - Means(j))/StdDeviations(j);
                end
            end
        end
        
        function Distances = euclideanDistance(NCC, sample, Centroids)
            % Distances = euclideanDistance(NCC, sample, Centroids)
            %
            % This function returns the distances calculations obtained with
            % Euclidean Distance method
            %
            % @param sample is an attribute vector to be classified
            % @param Centroids is a set of attribute vectors centroids to train the classifier
            % @return Distances a vector that represents the distance of 'sample' to the classes centroids
            %
            
            [rows cols] = size(Centroids);
            Distances = zeros(rows, 1);
            
            for i = 1:rows
                Distances(i) = sqrt(sum(minus(sample, Centroids(i, :)).^2));
            end
        end 
    end
end
