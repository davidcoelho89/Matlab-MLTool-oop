classdef ArtificialRegressionDataset
    % Hyperparameters
    properties
        name = " "
        nSamples = 200
        noiseStd = 0.1
        randomState = 1
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        X
        Y
        featureNames
        targetName
        description
    end
    
    methods
        function obj = ArtificialRegressionDataset(datasetName, varargin)

            if nargin < 1
                error("Informe o nome do dataset.");
            end

            % Parâmetros opcionais
            for i = 1:2:length(varargin)
                switch lower(varargin{i})
                    case "nsamples"
                        obj.nSamples = varargin{i+1};
                    case "noisestd"
                        obj.noiseStd = varargin{i+1};
                    case "randomstate"
                        obj.randomState = varargin{i+1};
                    otherwise
                        error("Parâmetro desconhecido: %s", varargin{i});
                end
            end

            rng(obj.randomState);

            obj.name = string(datasetName);

            switch lower(datasetName)
                
                case lower("LinearRegression")
                    obj = obj.generateLinearRegression();

                case lower("MultipleLinearRegression")
                    obj = obj.generateMultipleLinearRegression();

                case lower("PolynomialRegression")
                    obj = obj.generatePolynomialRegression();

                case lower("SinRegression")
                    obj = obj.generateSinRegression();

                case lower("SincRegression")
                    obj = obj.generateSincRegression();

                case lower("FrankeFunction")
                    obj = obj.generateFrankeFunction();

                otherwise
                    error("Dataset desconhecido: %s", datasetName);
            end
        end
    end
    
    methods (Access = private)
        
        function obj = generateLinearRegression(obj)
            
            x = linspace(0,10,obj.nSamples)';
            
            slope = 2;
            intercept = 1;
            
            y = slope*x + intercept + obj.noiseStd*randn(obj.nSamples,1);
            
            obj.X = x;
            obj.Y = y;
            
            obj.featureNames = "x";
            obj.targetName = "y";
            
            obj.description = ...
                "Artificial linear regression: y = 2*x + 1 + noise.";
            
        end

        function obj = generateMultipleLinearRegression(obj)

            beta = [3; -2; 1.5];
            intercept = 5;

            obj.X = randn(obj.nSamples, 3);
            obj.Y = intercept + obj.X * beta + obj.noiseStd * randn(obj.nSamples, 1);
            
            obj.featureNames = ["x1", "x2", "x3"];
            obj.targetName = "y";

            obj.description = ...
                "Artificial multiple linear regression: y = 5 + 3*x1 - 2*x2 + 1.5*x3 + noise.";
        end

        function obj = generatePolynomialRegression(obj)

            obj.X = linspace(-3, 3, obj.nSamples)';
            obj.Y = 2*obj.X.^2 - 3*obj.X + 1 + obj.noiseStd * randn(obj.nSamples, 1);

            obj.featureNames = "x";
            obj.targetName = "y";

            obj.description = ...
                "Artificial polynomial regression: y = 2*x^2 - 3*x + 1 + noise.";
        end

        function obj = generateSinRegression(obj)

            obj.X = linspace(0, 2*pi, obj.nSamples)';
            obj.Y = sin(obj.X) + obj.noiseStd * randn(obj.nSamples, 1);
            
            obj.featureNames = "x";
            obj.targetName = "y";

            obj.description = ...
                "Artificial sinusoidal regression: y = sin(x) + noise.";
        end

        function obj = generateSincRegression(obj)

            x = linspace(-10, 10, obj.nSamples)';
            y = ones(size(x));
            idx = x ~= 0;
            y(idx) = sin(x(idx)) ./ x(idx);
            y = y + obj.noiseStd * randn(obj.nSamples, 1);

            obj.X = x;
            obj.Y = y;
            obj.featureNames = "x";
            obj.targetName = "y";

            obj.description = ...
                "Artificial sinc regression: y = sin(x)/x + noise.";
        end

        function obj = generateFrankeFunction(obj)

            nSide = ceil(sqrt(obj.nSamples));

            x = linspace(0, 1, nSide);
            y = linspace(0, 1, nSide);

            [X1, X2] = meshgrid(x, y);

            X1 = X1(:);
            X2 = X2(:);

            x = [X1, X2];

            y = obj.franke(X1, X2);
            y = y + obj.noiseStd * randn(size(y));

            % Ajusta para ter exatamente nSamples
            obj.X = x(1:obj.nSamples, :);
            obj.Y = y(1:obj.nSamples, :);
            
            obj.featureNames = ["x1", "x2"];
            obj.targetName = "y";

            obj.description = ...
                "Artificial Franke function regression with two input variables.";
        end
    end

    methods (Static, Access = private)

        function z = franke(x, y)

            term1 = 0.75 * exp( ...
                -((9*x - 2).^2)/4 - ((9*y - 2).^2)/4 );

            term2 = 0.75 * exp( ...
                -((9*x + 1).^2)/49 - (9*y + 1)/10 );

            term3 = 0.5 * exp( ...
                -((9*x - 7).^2)/4 - ((9*y - 3).^2)/4 );

            term4 = -0.2 * exp( ...
                -(9*x - 4).^2 - (9*y - 7).^2 );

            z = term1 + term2 + term3 + term4;
        end
    end
    
end