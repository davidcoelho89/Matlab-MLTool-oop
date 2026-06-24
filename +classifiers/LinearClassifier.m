classdef (Abstract) LinearClassifier < mltoolbox.classifiers.BaseClassifier
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
            
            if size(Xb,2) ~= size(obj.W,2)
                error("Dimension mismatch.");
            end
            
            scores = Xb * obj.W;
            
            if ~isempty(obj.encoder)
                yhat = obj.encoder.inverse_transform(scores);
            else
                [~, yhat] = max(scores, [], 2);
            end
            
        end

    end % end methods

end % end class