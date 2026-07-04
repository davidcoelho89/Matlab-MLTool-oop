classdef (Abstract) BaseEstimator < handle
    
	% Hyperparameters
    properties
        addBias (1,1) logical = true
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        modelName (1,1) string = ""
        isTrained (1,1) logical = false
        nFeatures (1,1) double = 0
    end
    
	methods (Abstract)
        obj = fit(obj, X, y)
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
                error('Number of rows in X must match Number of rows in y.');
            end

            if any(isnan(X(:))) || any(isinf(X(:)))
                error('X must not contain NaN or Inf values.');
            end

            if any(isnan(y(:))) || any(isinf(y(:)))
                error('y must not contain NaN or Inf values.');
            end
        end
        
        function Xb = addBiasTerm(obj, X)
            if obj.addBias
                Xb = [ones(size(X,1),1), X];
            else
                Xb = X;
            end
        end
        
    end
    
end
