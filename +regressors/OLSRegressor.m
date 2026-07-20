classdef OLSRegressor < mltoolbox.baseModels.LinearRegressor

    % Hyperparameters
    properties
        approximation = 'pinv';     % | 'svd' | 'theoretical' |
        regularization = 0.0001;
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        P double = []
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
            
            obj.validateFitInputs(X,Y);

            obj.nFeatures = size(X,2);
            
            obj.nOutputs = size(Y,2);
            
            obj.isTrained = true;
            
            obj.P = (X'*X + obj.regularization*eye(obj.nFeatures)) \ eye(obj.nFeatures);
            
            switch obj.approximation
                case 'pinv'
                    obj.W = pinv(X) * Y;
                 case 'svd'
                    obj.W = X \ Y;
                case 'theoretical'
                    obj.W = (X'*X + obj.regularization*eye(obj.nFeatures)) \ (X'*Y);
                otherwise
                    obj.W = pinv(X) * Y;                    
            end
            
        end
        
        function obj = partial_fit(obj,x,y)
            
            if isempty(obj.W) || isempty(obj.P)
                obj.P = 1e+4 * eye(obj.nFeatures);
                obj.W = zeros(obj.nFeatures,obj.nOutputs);
            else
                K = obj.P*x'/(obj.lambda + x*obj.P*x');
                error = (y - x*obj.W);
                obj.W = obj.W + K*error;
                obj.P = (1/obj.lambda)*(obj.P - K*x*obj.P);
            end
            
        end
        
    end % end methods
    
end % end class