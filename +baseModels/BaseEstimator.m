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
        
    	function validateFitInputs(~, X, Y)
            if ~isnumeric(X) || ~isreal(X)
                error('X must be a real numeric matrix.');
            end

            if isempty(X) || ~ismatrix(X)
                error('X must be a non-empty 2D matrix with size [N x p].');
            end

            if isempty(Y) || ~ismatrix(Y)
                error('Y must be a non-empty 2D matrix with size [N x c].');
            end

            if size(X,1) ~= size(Y,1)
                error('Number of rows in X must match Number of rows in y.');
            end

            if any(isnan(X(:))) || any(isinf(X(:)))
                error('X must not contain NaN or Inf values.');
            end

            if any(isnan(Y(:))) || any(isinf(Y(:)))
                error('Y must not contain NaN or Inf values.');
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
