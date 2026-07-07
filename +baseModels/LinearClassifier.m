classdef (Abstract) LinearClassifier < mltoolbox.baseModels.BaseClassifier
    %
    % LINEARCLASSIFIER Abstract base class for linear classifiers
    %
    % Library Convetion:
    %   X : [N x p]
    %   y : [N x 1]
    %
    % N = number of samples
    % p = number of attributes
    %
    % Properties (Hyperparameters - for setting)
    %
    %   .
    %
    % Properties (Parameters - protected)
    %
    %   W = regression matrix [Nc x p] or [Nc x p+1]
    %
    % Methods (for external use)
    %
    %   yhat = predict(obj, X)	% Prediction Function
    %
    % Methods (protected)
    %
    %   .
    %
    % ----------------------------------------------------------------
    
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