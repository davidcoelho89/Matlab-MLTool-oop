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
        
    end
    
end
