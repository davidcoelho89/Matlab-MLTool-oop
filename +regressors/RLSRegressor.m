classdef RLSRegressor < mltoolbox.baseModels.LinearRegressor
    
    % Hyperparameters
    properties
        lambda (1,1) double = 1    % forgiving factor [0.9 to 1]
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        P double = []
    end
    
    methods
        
        % Constructor
        function obj = RLSRegressor(varargin)
            
            obj.modelName = "RLS Regressor";
            
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
            
            obj.P = 1e+4 * eye(obj.nFeatures);
            obj.W = zeros(obj.nFeatures,obj.nOutputs);
            
            for i = 1:size(X,1)
                x = X(i,:);
                y = Y(i,:);
                obj = partial_fit(obj,x,y);
            end
            
        end
        
        function obj = partial_fit(obj,x,y)
            
            K = obj.P*x'/(obj.lambda + x*obj.P*x');
            error = (y - x*obj.W);
            obj.W = obj.W + K*error;
            obj.P = (1/obj.lambda)*(obj.P - K*x*obj.P);
            
        end
        
    end
    
end