classdef (Abstract) BaseSystemIdentifier < handle
    
    % Hyperparameters
    properties
        outputLag
        inputLag
        errorLag
        includeCurrentInput = false	% include u[n] even if inputLag = 0
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        regressor
        nInputSignals
        nOutputSignals
        pastOutputs
        pastInputs
        pastErrors
    end
    
	methods 
        function obj = fit(obj, u, y)
            % Get signal information
            obj.nInputSignals = size(u,1);
            obj.nOutputSignals = size(y,2);
            
            % Fit Regression Model
            [Phi, Y] = obj.buildRegressors(u, y);
            obj.regressor.fit(Phi, Y);
            obj.isTrained = obj.regressor.isTrained;
            % ToDo - Update Memories
            % obj.pastInputs = u;
            % obj.pastOutputs = y;
            % obj.pastErrors = Y - obj.regressor.predict(Phi);
        end
        
        function yhat = predict(obj, u, y)
            obj.check_fitted();
            [Phi_test, ~] = obj.buildRegressors(u, y);
            yhat = obj.regressor.predict(Phi_test);            
        end
        
        function s = score(obj, u, y, method)
            % Calcula métricas diretamente a partir de sinais de entrada/saída
            % method: 'R2', 'R2adj', 'RMSE', 'MAE'
            yhat = obj.predictFromSignals(u, y);
            s = obj.score(yhat, y, method);
        end
        
    end
    
    methods (Access = protected)
        
        function obj = check_fitted(obj)
            % Sobrescreve check_fitted para verificar o regressor interno
            if isempty(obj.regressor) || ~obj.regressor.isTrained
                error("Model not fitted. Call trainFromSignals or fit first.");
            end
        end        
        
    end
    
end
















