classdef (Abstract) baseClassifier
    %BASECLASSIFIER Classe abstrata base para classificadores
    %
    % Convenção da biblioteca:
    %   X : [N x p]
    %   y : [N x 1]
    %
    % N = número de amostras
    % p = número de atributos

    properties
        addBias (1,1) logical = true
    end

    properties (SetAccess = protected)
        modelName (1,1) string = ""
        isTrained (1,1) logical = false
        classLabels = []
        nFeatures (1,1) double = 0
        nClasses (1,1) double = 0
    end

    methods (Abstract)
        obj = fit(obj, X, y)
        yhat = predict(obj, X)
    end

    methods
        function acc = score(obj, X, y)
            y = obj.normalizeLabels(y);
            yhat = obj.predict(X);
            acc = mean(yhat == y);
        end
    end

    methods (Access = protected)
        
        function validateFitInputs(obj, X, y)
            if ~isnumeric(X) || ~isreal(X)
                error('X must be a real numeric matrix.');
            end

            if isempty(X) || ndims(X) ~= 2
                error('X must be a non-empty 2D matrix with size [N x p].');
            end

            if isempty(y)
                error('y must not be empty.');
            end

            y = obj.normalizeLabels(y);

            if size(X,1) ~= numel(y)
                error('Number of rows in X must match length of y.');
            end

            if any(isnan(X(:))) || any(isinf(X(:)))
                error('X must not contain NaN or Inf values.');
            end

            if any(isnan(y(:))) || any(isinf(y(:)))
                error('y must not contain NaN or Inf values.');
            end
        end

        function validatePredictInput(obj, X)
            if ~obj.isTrained
                error('Model has not been trained yet.');
            end

            if ~isnumeric(X) || ~isreal(X)
                error('X must be a real numeric matrix.');
            end

            if isempty(X) || ndims(X) ~= 2
                error('X must be a non-empty 2D matrix with size [N x p].');
            end

            if size(X,2) ~= obj.nFeatures
                error('X must have %d columns (features).', obj.nFeatures);
            end

            if any(isnan(X(:))) || any(isinf(X(:)))
                error('X must not contain NaN or Inf values.');
            end
        end

        function y = normalizeLabels(~, y)
            if isrow(y)
                y = y.';
            end

            if ~isvector(y)
                error('y must be a label vector.');
            end
        end

        function Xb = addBiasTerm(obj, X)
            if obj.addBias
                Xb = [ones(size(X,1),1), X];
            else
                Xb = X;
            end
        end

        function [Yoh, classLabels] = oneHotEncodeLabels(~, y)
            classLabels = unique(y, 'stable');
            nClasses = numel(classLabels);
            N = numel(y);

            Yoh = zeros(N, nClasses);

            for k = 1:nClasses
                Yoh(:,k) = (y == classLabels(k));
            end
        end

        function yhat = decodeScores(~, scores, classLabels)
            [~, idx] = max(scores, [], 2);
            yhat = classLabels(idx);
        end
        
    end
end