classdef ArtificialSystemIdentificationDataset
    %ARTIFICIALSYSTEMIDENTIFICATIONDATASET Artificial data for system ID.
    %
    % Example:
    %   data = ArtificialSystemIdentificationDataset("ARX", ...
    %       "nSamples", 1000, "noiseStd", 0.05, "inputType", "prbs");
    %
    % Main outputs:
    %   data.U               Input signal (samples x inputs)
    %   data.Y               Output signal (samples x outputs)
    %   data.time            Time vector
    %   data.trueParameters  Parameters used to generate the data

    % Hyperparameters
    properties
        name = " "
        nSamples = 500
        noiseStd = 0.05
        randomState = 1
        sampleTime = 1
        inputType = "prbs"
    end

    % Generated data and metadata
    properties (GetAccess = public, SetAccess = protected)
        U
        Y
        time
        inputNames
        outputNames
        description
        trueParameters
    end

    methods
        function obj = ArtificialSystemIdentificationDataset(datasetName, varargin)
            if nargin < 1
                error("Informe o nome do dataset.");
            end

            if mod(length(varargin), 2) ~= 0
                error("Os parametros opcionais devem ser pares Name-Value.");
            end

            for i = 1:2:length(varargin)
                parameterName = lower(string(varargin{i}));

                switch parameterName
                    case "nsamples"
                        obj.nSamples = varargin{i+1};
                    case "noisestd"
                        obj.noiseStd = varargin{i+1};
                    case "randomstate"
                        obj.randomState = varargin{i+1};
                    case "sampletime"
                        obj.sampleTime = varargin{i+1};
                    case "inputtype"
                        obj.inputType = string(varargin{i+1});
                    otherwise
                        error("Parametro desconhecido: %s", string(varargin{i}));
                end
            end

            obj.validateParameters();
            rng(obj.randomState);

            obj.name = string(datasetName);
            obj.time = (0:obj.nSamples-1)' * obj.sampleTime;

            switch lower(string(datasetName))
                case "firstordersystem"
                    obj = obj.generateFirstOrderSystem();
                case "secondordersystem"
                    obj = obj.generateSecondOrderSystem();
                case "firsystem"
                    obj = obj.generateFIRSystem();
                case "arx"
                    obj = obj.generateARX();
                case "statespacesystem"
                    obj = obj.generateStateSpaceSystem();
                case "nonlinearnarx"
                    obj = obj.generateNonlinearNARX();
                case "hammersteinsystem"
                    obj = obj.generateHammersteinSystem();
                case "wienersystem"
                    obj = obj.generateWienerSystem();
                otherwise
                    error("Dataset desconhecido: %s", string(datasetName));
            end
        end
    end

    methods (Access = private)
        function validateParameters(obj)
            if ~isscalar(obj.nSamples) || obj.nSamples < 5 || ...
                    obj.nSamples ~= floor(obj.nSamples)
                error("nSamples deve ser um inteiro maior ou igual a 5.");
            end

            if ~isscalar(obj.noiseStd) || obj.noiseStd < 0
                error("noiseStd deve ser um escalar nao negativo.");
            end

            if ~isscalar(obj.sampleTime) || obj.sampleTime <= 0
                error("sampleTime deve ser um escalar positivo.");
            end

            validInputs = ["prbs", "whitenoise", "step", "sine", "chirp"];
            if ~any(strcmpi(obj.inputType, validInputs))
                error("inputType deve ser: prbs, whitenoise, step, sine ou chirp.");
            end
        end

        function obj = generateFirstOrderSystem(obj)
            % y(k) = 0.8*y(k-1) + 0.5*u(k-1) + e(k)
            obj.U = obj.generateInput(1);
            obj.Y = zeros(obj.nSamples, 1);

            a = 0.8;
            b = 0.5;
            for k = 2:obj.nSamples
                obj.Y(k) = a*obj.Y(k-1) + b*obj.U(k-1);
            end
            obj.Y = obj.addMeasurementNoise(obj.Y);

            obj.trueParameters = struct('a', a, 'b', b, 'delay', 1);
            obj.inputNames = "u";
            obj.outputNames = "y";
            obj.description = ...
                "First-order system: y(k) = 0.8*y(k-1) + 0.5*u(k-1) + e(k).";
        end

        function obj = generateSecondOrderSystem(obj)
            % Stable system with a pair of complex poles.
            obj.U = obj.generateInput(1);
            obj.Y = zeros(obj.nSamples, 1);

            a = [1.5, -0.7];
            b = [0.5, 0.2];
            for k = 3:obj.nSamples
                obj.Y(k) = a(1)*obj.Y(k-1) + a(2)*obj.Y(k-2) + ...
                    b(1)*obj.U(k-1) + b(2)*obj.U(k-2);
            end
            obj.Y = obj.addMeasurementNoise(obj.Y);

            obj.trueParameters = struct('a', a, 'b', b, 'delay', 1);
            obj.inputNames = "u";
            obj.outputNames = "y";
            obj.description = ...
                "Second-order system with two past outputs and two past inputs.";
        end

        function obj = generateFIRSystem(obj)
            % y(k) = b0*u(k) + b1*u(k-1) + b2*u(k-2) + e(k)
            obj.U = obj.generateInput(1);
            b = [0.7, -0.4, 0.2];
            obj.Y = filter(b, 1, obj.U);
            obj.Y = obj.addMeasurementNoise(obj.Y);

            obj.trueParameters = struct('b', b, 'delay', 0);
            obj.inputNames = "u";
            obj.outputNames = "y";
            obj.description = ...
                "FIR system: y(k) = 0.7*u(k) - 0.4*u(k-1) + 0.2*u(k-2) + e(k).";
        end

        function obj = generateARX(obj)
            % Standard ARX sanity check.
            obj.U = obj.generateInput(1);
            obj.Y = zeros(obj.nSamples, 1);

            a = [1.5, -0.7];
            b = [0.5, 0.2];
            processNoise = obj.noiseStd * randn(obj.nSamples, 1);

            for k = 3:obj.nSamples
                obj.Y(k) = a(1)*obj.Y(k-1) + a(2)*obj.Y(k-2) + ...
                    b(1)*obj.U(k-1) + b(2)*obj.U(k-2) + processNoise(k);
            end

            obj.trueParameters = struct('a', a, 'b', b, 'delay', 1);
            obj.inputNames = "u";
            obj.outputNames = "y";
            obj.description = ...
                "ARX system with na = 2, nb = 2, input delay = 1 and equation noise.";
        end

        function obj = generateStateSpaceSystem(obj)
            % Two-state, two-input and two-output stable system.
            A = [0.85, 0.10; -0.05, 0.75];
            B = [0.40, 0.10; 0.05, 0.30];
            C = eye(2);
            D = zeros(2);

            obj.U = obj.generateInput(2);
            obj.Y = zeros(obj.nSamples, 2);
            x = zeros(2, 1);

            for k = 1:obj.nSamples
                obj.Y(k,:) = (C*x + D*obj.U(k,:)')';
                x = A*x + B*obj.U(k,:)';
            end
            obj.Y = obj.addMeasurementNoise(obj.Y);

            obj.trueParameters = struct('A', A, 'B', B, 'C', C, 'D', D);
            obj.inputNames = ["u1", "u2"];
            obj.outputNames = ["y1", "y2"];
            obj.description = "Stable MIMO state-space system with two states.";
        end

        function obj = generateNonlinearNARX(obj)
            obj.U = obj.generateInput(1);
            obj.Y = zeros(obj.nSamples, 1);

            for k = 3:obj.nSamples
                obj.Y(k) = 0.7*obj.Y(k-1) - 0.2*obj.Y(k-2) + ...
                    0.1*obj.U(k-1) + 0.05*obj.U(k-1)^2;
            end
            obj.Y = obj.addMeasurementNoise(obj.Y);

            obj.trueParameters = struct('outputCoefficients', [0.7, -0.2], ...
                'inputCoefficient', 0.1, 'quadraticCoefficient', 0.05);
            obj.inputNames = "u";
            obj.outputNames = "y";
            obj.description = "Nonlinear NARX system with a quadratic input term.";
        end

        function obj = generateHammersteinSystem(obj)
            obj.U = obj.generateInput(1);
            v = obj.U + 0.5*obj.U.^3;
            obj.Y = zeros(obj.nSamples, 1);

            for k = 2:obj.nSamples
                obj.Y(k) = 0.8*obj.Y(k-1) + 0.2*v(k-1);
            end
            obj.Y = obj.addMeasurementNoise(obj.Y);

            obj.trueParameters = struct('linearA', 0.8, 'linearB', 0.2, ...
                'cubicCoefficient', 0.5, 'delay', 1);
            obj.inputNames = "u";
            obj.outputNames = "y";
            obj.description = ...
                "Hammerstein system: cubic static nonlinearity followed by linear dynamics.";
        end

        function obj = generateWienerSystem(obj)
            obj.U = obj.generateInput(1);
            x = zeros(obj.nSamples, 1);

            for k = 2:obj.nSamples
                x(k) = 0.8*x(k-1) + 0.2*obj.U(k-1);
            end
            obj.Y = tanh(x);
            obj.Y = obj.addMeasurementNoise(obj.Y);

            obj.trueParameters = struct('linearA', 0.8, 'linearB', 0.2, ...
                'outputNonlinearity', "tanh", 'delay', 1);
            obj.inputNames = "u";
            obj.outputNames = "y";
            obj.description = ...
                "Wiener system: linear dynamics followed by a tanh nonlinearity.";
        end

        function u = generateInput(obj, nInputs)
            u = zeros(obj.nSamples, nInputs);

            for j = 1:nInputs
                switch lower(obj.inputType)
                    case "prbs"
                        % Order-10 LFSR: x^10 + x^7 + 1.
                        % Implemented locally to avoid toolbox dependencies.
                        state = rand(1,10) >= 0.5;
                        if ~any(state)
                            state(end) = true;
                        end
                        for k = 1:obj.nSamples
                            u(k,j) = 2*double(state(end)) - 1;
                            feedback = xor(state(10), state(7));
                            state = [feedback, state(1:9)];
                        end
                    case "whitenoise"
                        u(:,j) = randn(obj.nSamples, 1);
                    case "step"
                        stepIndex = max(2, floor(obj.nSamples/4));
                        u(stepIndex:end,j) = 1;
                    case "sine"
                        frequency = (0.03 + 0.02*(j-1)) / obj.sampleTime;
                        u(:,j) = sin(2*pi*frequency*obj.time);
                    case "chirp"
                        % Linear chirp implemented without Signal Processing Toolbox.
                        duration = max(obj.time(end), obj.sampleTime);
                        f0 = 0.005 / obj.sampleTime;
                        f1 = 0.20 / obj.sampleTime;
                        chirpRate = (f1 - f0) / duration;
                        phase = 2*pi*(f0*obj.time + 0.5*chirpRate*obj.time.^2);
                        u(:,j) = sin(phase + (j-1)*pi/4);
                end
            end
        end

        function y = addMeasurementNoise(obj, y)
            y = y + obj.noiseStd * randn(size(y));
        end
    end
end