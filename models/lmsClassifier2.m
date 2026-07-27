classdef lmsClassifier2 < baseClassifier
    %LMSCLASSIFIER Classificador linear treinado com LMS
    %
    % Convenção:
    %   X : [N x p]
    %   y : [N x 1]

    properties
        learningRate (1,1) double = 0.01
        numEpochs (1,1) double = 200
        shuffleEachEpoch (1,1) logical = true
        weightInitScale (1,1) double = 0.01
        storeLossHistory (1,1) logical = true
    end

    properties (GetAccess = public, SetAccess = protected)
        W double = []
        lossHistory double = []
    end

    methods
        function obj = lmsClassifier2(varargin)
            
            obj.modelName = "LMS Classifier";
            
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

        function obj = fit(obj, X, y)
            
            obj.validateFitInputs(X, y);

            y = obj.normalizeLabels(y);
            
            [N, p] = size(X);
            [Yoh, classLabels] = obj.oneHotEncodeLabels(y);

            obj.nFeatures = p;
            obj.nClasses = size(Yoh, 2);
            obj.classLabels = classLabels;

            obj = obj.initializeWeights();

            if obj.storeLossHistory
                obj.lossHistory = zeros(obj.numEpochs, 1);
            else
                obj.lossHistory = [];
            end

            for ep = 1:obj.numEpochs
                if obj.shuffleEachEpoch
                    idx = randperm(N);
                else
                    idx = 1:N;
                end

                Xep = X(idx, :);
                Yep = Yoh(idx, :);

                for n = 1:N
                    obj = obj.partial_fit(Xep(n, :), Yep(n, :));
                end

                if obj.storeLossHistory
                    scores = obj.rawScores(X);
                    err = Yoh - scores;
                    obj.lossHistory(ep) = mean(sum(err.^2, 2));
                end
            end

            obj.isTrained = true;
        end

        function obj = partial_fit(obj, x, yOneHot)
            if iscolumn(x)
                x = x.';
            end

            if iscolumn(yOneHot)
                yOneHot = yOneHot.';
            end

            xb = obj.addBiasTerm(x);      % [1 x (p+1)] ou [1 x p]
            yhat = xb * obj.W;            % [1 x C]
            e = yOneHot - yhat;           % [1 x C]

            obj.W = obj.W + obj.learningRate * (xb.' * e);
        end

        function scores = predictScores(obj, X)
            obj.validatePredictInput(X);
            Xb = obj.addBiasTerm(X);
            scores = Xb * obj.W;
        end
        
        function yhat = predict(obj, X)
            scores = obj.predictScores(X);
            yhat = obj.decodeScores(scores, obj.classLabels);
        end
    end

    methods (Access = protected)
        function scores = rawScores(obj, X)
            Xb = obj.addBiasTerm(X);
            scores = Xb * obj.W;
        end
    end
    
    methods (Access = private)
        function obj = initializeWeights(obj)
            nRows = obj.nFeatures + double(obj.addBias);
            obj.W = obj.weightInitScale * randn(nRows, obj.nClasses);
        end
    end
end