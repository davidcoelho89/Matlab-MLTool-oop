classdef (Abstract) LinearRegressor < mltoolbox.baseModels.BaseRegressor
    
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