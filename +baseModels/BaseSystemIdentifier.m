classdef (Abstract) BaseSystemIdentifier < handle
    
    % Hyperparameters
    properties
        outputLag
        inputLag
        errorLag
        includeCurrentInput (1,1) logical = false
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        regressor
        nInputSignals
        nOutputSignals
        modelName (1,1) string = ""
        isTrained (1,1) logical = false
        pastOutputs
        pastInputs
        pastErrors
    end
    
	methods 
        function obj = fit(obj, u, y)
            
            % Get signal information
            obj.nInputSignals = size(u,2);
            obj.nOutputSignals = size(y,2);

            % Fit Regression Model
            % [Phi, Y] = obj.buildRegressors(u, y);
            % obj.regressor.fit(Phi, Y);
            % obj.isTrained = obj.regressor.isTrained;
            % obj.pastInputs = u;
            % obj.pastOutputs = y;
            % obj.pastErrors = Y - obj.regressor.predict(Phi);
            
        end
        
        function [Phi, Y] = buildRegressors(obj,u,y)
            
            lag_max = max([max(obj.outputLag)...
                           max(obj.inputLag)...
                           max(obj.errorLag)]);
            
            sum_of_lags = obj.outputLag * obj.nOutputSignals + ...
                          obj.inputLag * obj.nInputSignals + ...
                          obj.errorLag * obj.nOutputSignals ;
            
            signalsLength = size(u,1);
            
            Phi = zeros(signalsLength-lagMax, sum_of_lags);
            
            
            Y = y(1+lag_max:end,:);

        end
        
        function yhat = predict(obj, u, y)
            obj.check_fitted();
            [Phi_test, ~] = obj.buildRegressors(u, y);
            yhat = obj.regressor.predict(Phi_test);            
        end
        
        function s = score(obj, u, y, method)
            % method: 'R2', 'R2adj', 'RMSE', 'MAE'
            yhat = obj.predictFromSignals(u, y);
            s = obj.score(yhat, y, method);
        end
        
    end
    
    methods (Access = protected)
        
        function obj = check_fitted(obj)
            % Sobrescreve check_fitted para verificar o regressor interno
            if isempty(obj.regressor) || ~obj.regressor.isTrained
                error("Model not fitted. Call fit first.");
            end
        end        
        
    end
    
end
















