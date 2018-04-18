classdef KNN_Classifier
    % KNN_Classifier
    % 
    % This class is an implementation of the K-Nearest Neighbor Classifier
    %
    properties
        k_neighbors = 1;                   % The K neighbors to consider
        distance_method = "euclidean";     % The distance calculation method to use
        allowed_distances = ["euclidean"]; % Distance methods allowed implemented in this class
        use_zscore = true;                 % Whether to use or not the z-score normalization
        Distances;                         % Distances from the 'sample' to each vector of the 'Base'
    end
    
    methods
        function KNN = KNN_Classifier(k_neighbors, distance_method, use_zscore)
            % KNN = KNN_Classifier(k_neighbors, distance_method, use_zscore)
            %
            % This is the class construction function.
            %
            % @param k_neighbors is an integer value to K
            % @param distance_method is a string that must exists in allowed_distances
            % @param use_zscore is a boolean to choose whether zscore normalization should be used or not
            % @return KNN is a formatted object that will be returned
            %
            
            if (k_neighbors > 0)
                KNN.k_neighbors = k_neighbors;
            else
                fprintf("K = %d is invalid! Using K = 1.", k_neighbors);
            end
            
            if (size(find(KNN.allowed_distances == distance_method, 1, 'last'), 2))
                KNN.distance_method = distance_method;
            else
                fprintf("The selected distance '%s' could not be used. Using 'euclidean' instead.", distance_method);
            end
            
            KNN.use_zscore = use_zscore;
        end
        
        function class = classify(KNN, sample, Base, Classes)
            % class = classify(KNN, sample, Base, Classes)
            %
            % The main function of a classifier. The method classify return a
            % class index in order to classify the sent sample.
            %
            % @param sample is an attribute vector to be classified
            % @param Base is a set of attribute vectors to train the classifier
            % @param Classes is a set of classes assigned to each attribute vector in 'Base'. It is used to train the classifier, as well
            % @return class a scalar class index assigned to 'sample' as a classification result
            %
            
            KNN.Distances = KNN.calculateDistances(sample, Base);
            nearest_neighbors_indexes = KNN.findNearestNeighbours(KNN.Distances);
            
            votes = zeros(1, size(Classes, 2));
            for i = 1:KNN.k_neighbors
                votes = votes + Classes(nearest_neighbors_indexes(i), :);
            end
            
            [value1 index] = max(votes);
            class = index;
            
            if (mod(KNN.k_neighbors, 2) == 0)
                votes(index)  = 0;
                [value2 index] = max(votes);
                
                if (value1 == value2)
                    fprintf("There was a draw. Testing now with K-1.");
                    KNN.k_neighbors = KNN.k_neighbors - 1;
                    class = KNN.classify(sample, Base, Classes);
                end
            end
        end
        
        function neighbors_indexes = findNearestNeighbours(KNN, Distances)
            % neighbors_indexes = findNearestNeighbours(KNN, Distances)
            %
            % This method search for the nearest centroid to 'sample',
            % considering the given 'Distances'.
            %
            % @param Distances a vector that represents the distance of 'sample' to other classes samples
            % @return neighbors_indexes a vector with classes indexes of the K-nearest neightbors
            %
            
            neighbors_indexes = zeros(KNN.k_neighbors, 1);
            
            for i = 1:KNN.k_neighbors
                [value, index] = min(Distances);
                neighbors_indexes(i) = index;
                Distances(index) = [];
            end
        end
        
        function Distances = calculateDistances(KNN, sample, Base)
            % Distances = calculateDistances(KNN, sample, Base)
            %
            % This function calculate the distance from 'sample' to the classes
            % using a given distance calculation method and applying z-score to
            % data, if defined by the user.
            %
            % @param sample is an attribute vector to be classified
            % @param Base is a set of attribute vectors to train the classifier
            % @return Distances a vector that represents the distance of 'sample' to other classes samples
            %
            
            if (KNN.use_zscore)
                X = [sample; Base];
                Z = KNN.zscoreIt(X);
                sample = Z(1, :);
                Base   = Z(2:size(Z, 1), :);
            end
            
            switch (KNN.distance_method)
                case "euclidean"
                    Distances = KNN.euclideanDistance(sample, Base);
                otherwise
                    Distances = KNN.euclideanDistance(sample, Base);
            end
        end
        
        function Z = zscoreIt(KNN, X)
            % Z = zscoreIt(KNN, X)
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
        
        function Distances = euclideanDistance(KNN, sample, Base)
            % Distances = euclideanDistance(KNN, sample, Base)
            %
            % This function returns the distances calculations obtained with
            % Euclidean Distance method
            %
            % @param sample is an attribute vector to be classified
            % @param Base is a set of attribute vectors to train the classifier
            % @return Distances a vector that represents the distance of 'sample' to other classes samples
            %
            
            [rows cols] = size(Base);
            Distances = zeros(rows, 1);
            
            for i = 1:rows
                Distances(i) = sqrt(sum(minus(sample, Base(i, :)).^2));
            end
        end 
    end
end
