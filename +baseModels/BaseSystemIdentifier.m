classdef (Abstract) BaseSystemIdentifier < handle
    
    % Hyperparameters
    properties
        outputLags = []
        inputLags = []
        errorLags = []
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
            
            obj.nInputSignals = size(u,2);
            obj.nOutputSignals = size(y,2);
            [Phi, Y] = obj.buildRegressors(u, y);
            obj.regressor.fit(Phi, Y);
            obj.isTrained = obj.regressor.isTrained;

            % ToDo - Get pastOutputs
            % ToDo - Get pastInputs
            % ToDo - Get pastErrors
            
        end
        
        function [Phi, Y] = buildRegressors(obj,u,y)
            
            lag_max = max([max(obj.outputLags)...
                           max(obj.inputLags)...
                           max(obj.errorLags)]);
            disp(lag_max);
            
            sum_of_lags = sum([sum(obj.outputLags) ...
                               sum(obj.inputLags) ...
                               sum(obj.errorLags) ...
                               obj.includeCurrentInput*obj.nInputSignals]) ;
            disp(sum_of_lags);
            
            signalsLength = size(u,1);
            
            Phi = zeros(signalsLength-lag_max, sum_of_lags);
            disp(size(Phi));
            
            Y = y(1+lag_max:end,:);
            disp(size(Y));
            
            cont = 0;
            
            % Add outputs to Phi matrix
            for outputSignal = 1:obj.nOutputSignals
                outputLag = obj.outputLags(outputSignal);
                for j = 1:outputLag
                    cont = cont + 1;
                    Phi(:,cont) = y(1+lag_max-j:end-j,outputSignal);
                end
            end
            
            % Add inputs to Phi matrix
            for inputSignal = 1:obj.nInputSignals
                if(obj.includeCurrentInput)
                    cont = cont + 1;
                    Phi(:,cont) = u(1+lag_max:end,inputSignal);
                end
                inputLag = obj.inputLags(inputSignal);
                for j = 1:inputLag
                    cont = cont + 1;
                    Phi(:,cont) = u(1+lag_max-j:end-j,inputSignal);
                end
            end            

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
















