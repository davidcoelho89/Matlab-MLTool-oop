classdef (Abstract) LinearRegressor < mltoolbox.baseModels.BaseRegressor
    %
    % LINEARREGRESSOR Abstract base class for linear regressors
    %
    % Library Convetion:
    %   X : [N x p]
    %   y : [N x 1]
    %
    % N = number of samples
    % p = number of attributes
    %
    % Properties (Hyperparameters - for setting)
    %
    %   .
    %
    % Properties (Parameters - protected)
    %
    %   W = regression matrix [Ny x p] or [Ny x p+1]
    %
    % Methods (for external use)
    %
    %   yhat = predict(obj, X)	% Prediction Function
    %
    % Methods (protected)
    %
    %   .
    %
    % ----------------------------------------------------------------
    
    % Hyperparameters
    properties
        
    end
    
	% Parameters
    properties (GetAccess = public, SetAccess = protected)
       W double = []   
    end
    
    methods
        
        function yhat = predict(obj, X)
            
            Xb = obj.addBiasTerm(X);
            
            obj.validatePredictInput(Xb);
            
            obj.validateEstimationMatrix(Xb);
            
            yhat = Xb * obj.W;
            
        end
        
    end
    
    methods (Access = protected)
        
        function validateEstimationMatrix(obj, X)
            if size(X,2) ~= size(obj.W,1)
                error("There is a Dimension mismatch between X and W.");
            end
        end
        
    end
    
end