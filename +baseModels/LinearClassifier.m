classdef (Abstract) LinearClassifier < mltoolbox.baseModels.BaseClassifier
    
    % Hyperparameters
    properties
        encoder = [];
    end

    % Parameters
    properties (GetAccess = public, SetAccess = protected)
       W double = []   
    end

    methods

        % Prediction Function
        function yhat = predict(obj, X)
            
            Xb = obj.addBiasTerm(X);

            obj.validatePredictInput(Xb);
            
            obj.validateEstimationMatrix(Xb);
            
            y_hat_mult = Xb * obj.W;
            
            if ~isempty(obj.encoder)
                yhat = obj.encoder.inverse_transform(y_hat_mult);
            else
                [~, yhat] = max(y_hat_mult, [], 2);
            end
            
        end

    end % end methods
    
    methods (Access = protected)
        
        function validateEstimationMatrix(obj, X)
            if size(X,2) ~= size(obj.W,1)
                error("There is a Dimension mismatch between X and W.");
            end
        end
        
    end
    

end % end class