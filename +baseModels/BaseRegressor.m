classdef (Abstract) BaseRegressor < mltoolbox.baseModels.BaseEstimator
    
    % Hyperparameters
    properties
        % the only hyperparameter is addBias (from baseEstimator)
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        nOutputs (1,1) double = 0
    end
    
    methods
        function r2 = score(obj, X, y, method)
            
            obj.validateFitInputs(X, y);
            obj.check_fitted();
            
            if nargin < 4
                method = 'r2';
            end
            
            yhat = obj.predict(X);
            n = numel(y);
            
            switch lower(method)
                case 'r2'
                    SSres = sum((y - yhat).^2);
                    SStot = sum((y - mean(y)).^2);
                    r2 = 1 - SSres/SStot;
                case 'r2adj'
                    SSres = sum((y - yhat).^2);
                    SStot = sum((y - mean(y)).^2);
                    p = obj.nFeatures;  % número de features
                    r2 = 1 - (1 - (1 - SSres/SStot))*(n-1)/(n-p-1);
                case 'rmse'
                    r2 = sqrt(mean((y - yhat).^2));
                case 'mae'
                    r2 = mean(abs(y - yhat));
                otherwise
                    error('Unknown score method.');
            end
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
    end
    
end