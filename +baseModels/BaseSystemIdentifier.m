classdef (Abstract) BaseSystemIdentifier < handle
    
    % Hyperparameters
    properties
        outputLags = []
        inputLags = []
        errorLags = []
        includeCurrentInput (1,1) logical = false
        stepsAhead = 1
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        regressor
        nInputSignals
        nOutputSignals
        modelName (1,1) string = ""
        isTrained (1,1) logical = false
        outputsMemory
        inputsMemory
        errorsMemory
    end
    
	methods 
        function obj = fit(obj, U, Y)
            
            obj.nInputSignals = size(U,2);
            obj.nOutputSignals = size(Y,2);

            [Phi, Y] = obj.buildRegressorsFromSignals(U, Y);
            
            obj.holdLastOutputsFromTraining(Y);
            obj.holdLastInputsFromTraining(U);
            
            obj.regressor.fit(Phi, Y);

            obj.isTrained = obj.regressor.isTrained;
        end
        
        function [Phi, Y] = buildRegressorsFromSignals(obj,Usig,Ysig)
            
            lag_max = max([max(obj.outputLags)...
                           max(obj.inputLags)...
                           max(obj.errorLags)]);
            disp(lag_max);
            
            sum_of_lags = sum([sum(obj.outputLags) ...
                               sum(obj.inputLags) ...
                               sum(obj.errorLags) ...
                               obj.includeCurrentInput*obj.nInputSignals]) ;
            disp(sum_of_lags);
            
            signalsLength = size(Usig,1);
            
            Phi = zeros(signalsLength-lag_max, sum_of_lags);
            disp(size(Phi));
            
            Y = Ysig(1+lag_max:end,:);
            disp(size(Y));
            
            cont = 0;
            
            % Add outputs to Phi matrix
            for outputSignal = 1:obj.nOutputSignals
                outputLag = obj.outputLags(outputSignal);
                for j = 1:outputLag
                    cont = cont + 1;
                    Phi(:,cont) = Ysig(1+lag_max-j:end-j,outputSignal);
                end
            end
            
            % Add inputs to Phi matrix
            for inputSignal = 1:obj.nInputSignals
                if(obj.includeCurrentInput)
                    cont = cont + 1;
                    Phi(:,cont) = Usig(1+lag_max:end,inputSignal);
                end
                inputLag = obj.inputLags(inputSignal);
                for j = 1:inputLag
                    cont = cont + 1;
                    Phi(:,cont) = Usig(1+lag_max-j:end-j,inputSignal);
                end
            end            

        end
        
        function obj = holdLastOutputsFromTraining(obj,Y)
            max_outputLag = max(obj.outputLags);
            obj.outputsMemory = flipud( Y(end - max_outputLag + 1 : end, :) );
        end
        
        function obj = holdLastInputsFromTraining(obj,U)
            max_inputLag = max(obj.inputLags);
            obj.inputsMemory = flipud ( U(end - max_inputLag + 1 : end, :) );
        end
        
        function yhat = predict(obj, U, Y)
            
            obj.check_fitted();
            
            signalsLength = size(U,1);
            yhat = zeros(signalsLength,obj.nOutputSignals);
            
            for i = 1:signalsLength
                u = U(i,:);
                y = Y(i,:);
                phi = obj.buildRegressorFromMemory(u);
                yhat(i,:) = obj.regressor.predict(phi);
                obj.updateInputsMemory(u);
                obj.updateOutputsMemory(y);
            end
            
        end
        
        function phi = buildRegressorFromMemory(obj,u)
            sum_of_lags = sum([sum(obj.outputLags) ...
                               sum(obj.inputLags) ...
                               sum(obj.errorLags) ...
                               obj.includeCurrentInput*obj.nInputSignals]) ;
            phi = zeros(1,sum_of_lags);
            
            init = 0;
            
            % Add outputs to phi vector
            for outputSignal = 1:obj.nOutputSignals
                 outputLag = obj.outputLags(outputSignal);
                 phi(init+1:init+outputLag) = ...
                     obj.outputsMemory(1:outputLag,outputSignal)';
                 init = init + outputLag;
            end
            
            % Add inputs to phi vector
            for inputSignal = 1:obj.nInputSignals
                if(obj.includeCurrentInput)
                    init = init+1;
                    phi(init) = u(inputSignal);
                end
                inputLag = obj.inputLags(inputSignal);
                phi(init+1:init+inputLag) = ...
                    obj.inputsMemory(1:inputLag,inputSignal)';
                init = init + inputLag;
            end
            
        end
        
        function obj = updateOutputsMemory(obj,y)
            obj.outputsMemory = [y ; obj.outputsMemory(1:end-1,:)];
        end
        
        function obj = updateInputsMemory(obj,u)
            obj.inputsMemory = [u ; obj.inputsMemory(1:end-1,:)];
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
















