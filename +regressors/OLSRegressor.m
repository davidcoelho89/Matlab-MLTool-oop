classdef OLSRegressor < mltoolbox.baseModels.LinearRegressor

    % Hyperparameters
    properties
        approximation = 'pinv';     % | 'svd' | 'theoretical' |
        regularization = 0.0001;
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        
    end
    
    methods
        
        % Constructor
        function obj = OLSRegressor(varargin)
            
            obj.modelName = "OLS Regressor";
            
            if mod(nargin,2) ~= 0
                error('Arguments must be given as name-value pairs.');
            end
            
           for k = 1:2:nargin
                name = varargin{k};
                value = varargin{k+1};

                if ~isprop(obj, name)
                    error('Unknown property: %s', string(name));
                end

                obj.(name) = value;
            end 
            
        end
        
        % Training Function (N instances)
        function obj = fit(obj,X,Y)
            
            X = obj.addBiasTerm(X);
            
            obj.validateFitInputs(X,Y)

            obj.nFeatures = size(X,2);
            
            obj.nOutputs = size(Y,2);
            
            switch obj.approximation
                case 'pinv'
                    obj.W = pinv(X) * Y;
                 case 'svd'
                    obj.W = X \ Y;
                case 'theoretical'
                    p = size(X,2);
                    obj.W = (X'*X + obj.regularization*eye(p)) \ (X'*Y);
                otherwise
                    obj.W = pinv(X) * Y;                    
            end
            
            obj.isTrained = true;
            
        end
        
    end % end methods
    
end % end class