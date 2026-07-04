classdef (Abstract) BaseClassifier < handle
    
    % Hyperparameters
    properties
        addBias (1,1) logical = true
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        modelName (1,1) string = ""
        isTrained (1,1) logical = false
        classLabels = [];
        nFeatures (1,1) double = 0
        nClasses (1,1) double = 0
    end

    methods (Abstract)
        obj = fit(obj, X, y)
    end

    methods
        function acc = score(obj, X, y)
            yhat = obj.predict(X);
            acc = mean(yhat == y);
        end
    end

    methods (Access = protected)
        
        function check_is_fitted(obj)
            if ~obj.isTrained
                error("Model not fitted. Call fit() first.");
            end
        end
        
        function validateFitInputs(~, X, y)
            if ~isnumeric(X) || ~isreal(X)
                error('X must be a real numeric matrix.');
            end

            if isempty(X) || ~ismatrix(X)
                error('X must be a non-empty 2D matrix with size [N x p].');
            end

            if isempty(y)
                error('y must not be empty.');
            end

            if size(X,1) ~= size(y,1)
                error('Number of rows in X must match length of y.');
            end

            if any(isnan(X(:))) || any(isinf(X(:)))
                error('X must not contain NaN or Inf values.');
            end

            if any(isnan(y(:))) || any(isinf(y(:)))
                error('y must not contain NaN or Inf values.');
            end
        end

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

        function Xb = addBiasTerm(obj, X)
            if obj.addBias
                Xb = [ones(size(X,1),1), X];
            else
                Xb = X;
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