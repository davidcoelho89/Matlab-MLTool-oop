classdef ArtificialClassificationDataset
    %ARTIFICIALCLASSIFICATIONDATASET Artificial datasets for classification.
    %
    %   The generated data follow the convention:
    %       inputs  : p x N (p features and N samples)
    %       outputs : 1 x N (integer class labels from 1 to nClasses)
    %
    %   Example:
    %       data = ArtificialClassificationDataset("GaussianBlobs", ...
    %           "nSamples", 300, "nClasses", 3, "nFeatures", 2, ...
    %           "noiseStd", 0.5, "randomState", 1);

    % Hyperparameters
    properties
        name = " "
        nSamples = 200
        nClasses = 2
        nFeatures = 2
        noiseStd = 0.1
        randomState = 1
    end

    % Generated dataset
    properties (GetAccess = public, SetAccess = protected)
        inputs
        outputs
        featureNames
        classNames
        description
    end

    methods
        function obj = ArtificialClassificationDataset(datasetName, varargin)

            if nargin < 1
                error("Informe o nome do dataset.");
            end

            if mod(length(varargin), 2) ~= 0
                error("Os parametros opcionais devem ser pares Name-Value.");
            end

            % Optional parameters
            for i = 1:2:length(varargin)
                parameterName = lower(string(varargin{i}));

                switch parameterName
                    case "nsamples"
                        obj.nSamples = varargin{i+1};
                    case "nclasses"
                        obj.nClasses = varargin{i+1};
                    case "nfeatures"
                        obj.nFeatures = varargin{i+1};
                    case "noisestd"
                        obj.noiseStd = varargin{i+1};
                    case "randomstate"
                        obj.randomState = varargin{i+1};
                    otherwise
                        error("Parametro desconhecido: %s", varargin{i});
                end
            end

            obj.validateParameters();
            rng(obj.randomState);

            obj.name = string(datasetName);

            switch lower(string(datasetName))
                case "gaussianblobs"
                    obj = obj.generateGaussianBlobs();

                case "linearclassification"
                    obj = obj.generateLinearClassification();

                case "moons"
                    obj = obj.generateMoons();

                case "circles"
                    obj = obj.generateCircles();

                case "xor"
                    obj = obj.generateXOR();

                case "spirals"
                    obj = obj.generateSpirals();

                otherwise
                    error("Dataset desconhecido: %s", datasetName);
            end

            obj = obj.shuffleSamples();
            obj = obj.setMetadata();
        end
    end

    methods (Access = private)
        function validateParameters(obj)
            if ~isscalar(obj.nSamples) || obj.nSamples < 1 || ...
                    obj.nSamples ~= floor(obj.nSamples)
                error("nSamples deve ser um inteiro positivo.");
            end

            if ~isscalar(obj.nClasses) || obj.nClasses < 2 || ...
                    obj.nClasses ~= floor(obj.nClasses)
                error("nClasses deve ser um inteiro maior ou igual a 2.");
            end

            if obj.nSamples < obj.nClasses
                error("nSamples deve ser maior ou igual a nClasses.");
            end

            if ~isscalar(obj.nFeatures) || obj.nFeatures < 1 || ...
                    obj.nFeatures ~= floor(obj.nFeatures)
                error("nFeatures deve ser um inteiro positivo.");
            end

            if ~isscalar(obj.noiseStd) || obj.noiseStd < 0
                error("noiseStd deve ser um escalar nao negativo.");
            end
        end

        function obj = generateGaussianBlobs(obj)
            % Random Gaussian clusters in a p-dimensional space.
            counts = obj.samplesPerClass();
            centers = 4 * randn(obj.nFeatures, obj.nClasses);

            obj.inputs = zeros(obj.nFeatures, obj.nSamples);
            obj.outputs = zeros(1, obj.nSamples);

            first = 1;
            clusterStd = obj.noiseStd;

            for c = 1:obj.nClasses
                last = first + counts(c) - 1;
                obj.inputs(:, first:last) = centers(:, c) + ...
                    clusterStd * randn(obj.nFeatures, counts(c));
                obj.outputs(first:last) = c;
                first = last + 1;
            end

            obj.description = ...
                "Gaussian clusters with configurable classes and features.";
        end

        function obj = generateLinearClassification(obj)
            % Classes separated mainly along the first feature.
            counts = obj.samplesPerClass();
            obj.inputs = zeros(obj.nFeatures, obj.nSamples);
            obj.outputs = zeros(1, obj.nSamples);

            classPositions = linspace(-2, 2, obj.nClasses);
            first = 1;

            for c = 1:obj.nClasses
                last = first + counts(c) - 1;
                x = randn(obj.nFeatures, counts(c));
                x(1, :) = classPositions(c) + obj.noiseStd * x(1, :);

                if obj.nFeatures > 1
                    x(2:end, :) = 0.5 * x(2:end, :);
                end

                obj.inputs(:, first:last) = x;
                obj.outputs(first:last) = c;
                first = last + 1;
            end

            obj.description = ...
                "Linearly separable classes, primarily along the first feature.";
        end

        function obj = generateMoons(obj)
            obj.requireTwoClassesAndFeatures("Moons", 2, 2);

            counts = obj.samplesPerClass();
            t1 = pi * rand(1, counts(1));
            t2 = pi * rand(1, counts(2));

            moon1 = [cos(t1); sin(t1)];
            moon2 = [1 - cos(t2); 0.5 - sin(t2)];

            obj.inputs = [moon1, moon2] + ...
                obj.noiseStd * randn(2, obj.nSamples);
            obj.outputs = [ones(1, counts(1)), 2 * ones(1, counts(2))];
            obj.description = "Two interleaving half-moon classes.";
        end

        function obj = generateCircles(obj)
            obj.requireTwoClassesAndFeatures("Circles", 2, 2);

            counts = obj.samplesPerClass();
            t1 = 2*pi * rand(1, counts(1));
            t2 = 2*pi * rand(1, counts(2));

            outerCircle = [cos(t1); sin(t1)];
            innerCircle = 0.45 * [cos(t2); sin(t2)];

            obj.inputs = [outerCircle, innerCircle] + ...
                obj.noiseStd * randn(2, obj.nSamples);
            obj.outputs = [ones(1, counts(1)), 2 * ones(1, counts(2))];
            obj.description = "Two concentric circular classes.";
        end

        function obj = generateXOR(obj)
            obj.requireTwoClassesAndFeatures("XOR", 2, 2);

            x = 2 * rand(2, obj.nSamples) - 1;
            y = 1 + double((x(1, :) >= 0) == (x(2, :) >= 0));

            obj.inputs = x + obj.noiseStd * randn(2, obj.nSamples);
            obj.outputs = y;
            obj.description = "Two-dimensional nonlinear XOR problem.";
        end

        function obj = generateSpirals(obj)
            if obj.nFeatures ~= 2
                error("Spirals requer nFeatures = 2.");
            end

            counts = obj.samplesPerClass();
            obj.inputs = zeros(2, obj.nSamples);
            obj.outputs = zeros(1, obj.nSamples);
            first = 1;

            for c = 1:obj.nClasses
                last = first + counts(c) - 1;
                radius = linspace(0.1, 1, counts(c));
                angle = 1.75 * 2*pi * radius + ...
                    2*pi*(c - 1)/obj.nClasses;

                points = [radius .* cos(angle); radius .* sin(angle)];
                points = points + obj.noiseStd * randn(2, counts(c));

                obj.inputs(:, first:last) = points;
                obj.outputs(first:last) = c;
                first = last + 1;
            end

            obj.description = "Interleaving spiral classes.";
        end

        function counts = samplesPerClass(obj)
            counts = floor(obj.nSamples / obj.nClasses) * ...
                ones(1, obj.nClasses);
            remainder = mod(obj.nSamples, obj.nClasses);
            counts(1:remainder) = counts(1:remainder) + 1;
        end

        function requireTwoClassesAndFeatures(obj, datasetName, classes, features)
            if obj.nClasses ~= classes || obj.nFeatures ~= features
                error("%s requer nClasses = %d e nFeatures = %d.", ...
                    datasetName, classes, features);
            end
        end

        function obj = shuffleSamples(obj)
            order = randperm(obj.nSamples);
            obj.inputs = obj.inputs(:, order);
            obj.outputs = obj.outputs(:, order);
        end

        function obj = setMetadata(obj)
            obj.featureNames = strings(1, obj.nFeatures);
            for j = 1:obj.nFeatures
                obj.featureNames(j) = "x" + j;
            end

            obj.classNames = strings(1, obj.nClasses);
            for c = 1:obj.nClasses
                obj.classNames(c) = "class" + c;
            end
        end
    end
end
