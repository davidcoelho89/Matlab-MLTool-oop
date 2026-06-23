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
        
    end

    % Parameters
    properties (GetAccess = public, SetAccess = protected)
       W double = []   
    end

    methods

        % Prediction Function
        function yhat = predict(obj, X)
            
            Xb = addBiasTerm(obj, X);
            
            scores = Xb * obj.W;
            
            % multiclass decision rule (argmax)
            [~, idx] = max(scores, [], 2);

            yhat = idx;
        end

    end % end methods

end % end class