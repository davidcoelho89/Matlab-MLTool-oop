classdef SystemIdentificationMetrics
    %SYSTEMIDENTIFICATIONMETRICS Accuracy and residual diagnostics.
    %   Samples are rows. Outputs and inputs are columns.

    methods (Static)
        function metrics = calculate(yTrue, yPred, u, varargin)
            if nargin < 3
                u = [];
            end

            p = inputParser;
            addParameter(p, 'MaxLag', []);
            addParameter(p, 'Normalization', 'range');
            addParameter(p, 'NumParameters', []);
            addParameter(p, 'SignificanceLevel', 0.05);
            parse(p, varargin{:});

            regression = RegressionMetrics.calculate(yTrue, yPred, ...
                'Normalization', p.Results.Normalization, ...
                'NumParameters', p.Results.NumParameters);

            if isrow(yTrue), yTrue = yTrue(:); end
            if isrow(yPred), yPred = yPred(:); end
            residuals = yTrue - yPred;
            nSamples = size(residuals, 1);
            nOutputs = size(residuals, 2);

            maxLag = p.Results.MaxLag;
            if isempty(maxLag)
                maxLag = min(20, nSamples - 1);
            end
            validateattributes(maxLag, {'numeric'}, ...
                {'scalar', 'integer', 'nonnegative', '<', nSamples});

            alpha = p.Results.SignificanceLevel;
            validateattributes(alpha, {'numeric'}, ...
                {'scalar', '>', 0, '<', 1});

            autocorrelation = zeros(2 * maxLag + 1, nOutputs);
            lags = (-maxLag:maxLag).';
            Q = NaN(1, nOutputs);
            pValue = NaN(1, nOutputs);

            for outputIndex = 1:nOutputs
                e = residuals(:, outputIndex);
                autocorrelation(:, outputIndex) = ...
                    SystemIdentificationMetrics.normalizedCorrelation(e, e, maxLag);

                positiveACF = autocorrelation(maxLag + 2:end, outputIndex);
                positiveLags = 1:maxLag;
                if maxLag > 0
                    Q(outputIndex) = nSamples * (nSamples + 2) * ...
                        sum((positiveACF.'.^2) ./ (nSamples - positiveLags));
                    degreesOfFreedom = maxLag;
                    pValue(outputIndex) = gammainc(Q(outputIndex) / 2, ...
                        degreesOfFreedom / 2, 'upper');
                end
            end

            zCritical = -sqrt(2) * erfcinv(2 * (1 - alpha / 2));
            confidenceLimit = zCritical / sqrt(nSamples);

            metrics.regression = regression;
            metrics.residuals.values = residuals;
            metrics.residuals.mean = mean(residuals, 1);
            metrics.residuals.std = std(residuals, 0, 1);
            metrics.residuals.lags = lags;
            metrics.residuals.autocorrelation = autocorrelation;
            metrics.residuals.confidenceLimit = confidenceLimit;
            metrics.residuals.ljungBoxQ = Q;
            metrics.residuals.ljungBoxPValue = pValue;
            metrics.residuals.isWhite = pValue > alpha;

            metrics.inputResidual = [];
            if ~isempty(u)
                validateattributes(u, {'numeric'}, ...
                    {'2d', 'nonempty', 'real', 'finite'});
                if isrow(u), u = u(:); end
                if size(u, 1) ~= nSamples
                    error('SystemIdentificationMetrics:InputSizeMismatch', ...
                        'u must have the same number of rows as yTrue.');
                end

                nInputs = size(u, 2);
                crossCorrelation = zeros(2 * maxLag + 1, nInputs, nOutputs);
                for inputIndex = 1:nInputs
                    for outputIndex = 1:nOutputs
                        crossCorrelation(:, inputIndex, outputIndex) = ...
                            SystemIdentificationMetrics.normalizedCorrelation( ...
                            u(:, inputIndex), residuals(:, outputIndex), maxLag);
                    end
                end
                metrics.inputResidual.lags = lags;
                metrics.inputResidual.crossCorrelation = crossCorrelation;
                metrics.inputResidual.confidenceLimit = confidenceLimit;
                metrics.inputResidual.isUncorrelated = squeeze(all( ...
                    abs(crossCorrelation) <= confidenceLimit, 1));
            end
        end
    end

    methods (Static, Access = private)
        function correlation = normalizedCorrelation(x, y, maxLag)
            x = x - mean(x);
            y = y - mean(y);
            denominator = sqrt(sum(x.^2) * sum(y.^2));
            correlation = NaN(2 * maxLag + 1, 1);
            if denominator == 0
                return;
            end

            lags = -maxLag:maxLag;
            for index = 1:numel(lags)
                lag = lags(index);
                if lag >= 0
                    correlation(index) = ...
                        sum(x(1 + lag:end) .* y(1:end - lag)) / denominator;
                else
                    positiveLag = -lag;
                    correlation(index) = ...
                        sum(x(1:end - positiveLag) .* y(1 + positiveLag:end)) / denominator;
                end
            end
        end
    end
end
