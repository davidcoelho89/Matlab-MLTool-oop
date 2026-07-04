classdef (Abstract) BaseClassifier < mltoolbox.baseModels.BaseEstimator
    
    % Hyperparameters
    properties
        % the only hyperparameter is addBias (from baseEstimator)
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        classLabels = [];
        nClasses (1,1) double = 0
    end

    methods
        function acc = score(obj, X, y)
            
            obj.validateFitInputs(X, y);
            obj.check_fitted();
            yhat = obj.predict(X);
            acc = mean(yhat == y);
            
        end
    end

    methods (Access = protected)
        
        function validatePredictInput(obj, X)
            
            obj.check_is_fitted();

            if ~isnumeric(X) || ~isreal(X)
                error('X must be a real numeric matrix.');
            end

            if isempty(X) || ~ismatrix(X)
                error('X must be a non-empty 2D matrix with size [N x p].');
            end
            
            if size(X,2) ~= obj.nFeatures
                error('X must have %d columns (features).', obj.nFeatures);
            end

            if any(isnan(X(:))) || any(isinf(X(:)))
                error('X must not contain NaN or Inf values.');
            end
        end

      	function [Yoh, classLabels] = oneHotEncodeLabels(~, y)
            
            classLabels = unique(y, 'stable');
            n = length(y);
            K = length(classLabels);
            
            Yoh = zeros(n,K);

            for k = 1:K
                Yoh(:,k) = (y == classLabels(k));
            end
        end

        function yhat = decodeScores(obj, scores)
            
            obj.check_is_fitted();
            
            if isempty(obj.classLabels)
                error("classLabels not initialized. Call fit().");
            end
            
            [~, idx] = max(scores, [], 2);
            yhat = obj.classLabels(idx);
            
        end
        
    end
end