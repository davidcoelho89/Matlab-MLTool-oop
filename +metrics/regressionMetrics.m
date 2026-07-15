classdef regressionMetrics

    methods (Static)
        function metrics = calculate(yTrue, yPred, varargin)
            p = inputParser;
            addParameter(p, 'Normalization', 'range');
            addParameter(p, 'NumParameters', []);
            parse(p, varargin{:});

            [yTrue, yPred] = mltoolbox.metrics.regressionMetrics.validateSignals(yTrue, yPred);
            normalization = lower(char(p.Results.Normalization));
            nParameters = p.Results.NumParameters;

            validNormalizations = {'range', 'std', 'mean'};
            if ~any(strcmp(normalization, validNormalizations))
                error('regressionMetrics:InvalidNormalization', ...
                    'Normalization must be ''range'', ''std'', or ''mean''.');
            end

            if ~isempty(nParameters)
                validateattributes(nParameters, {'numeric'}, ...
                    {'scalar', 'integer', 'nonnegative'});
            end

            nSamples = size(yTrue, 1);
            nOutputs = size(yTrue, 2);
            residuals = yTrue - yPred;

            SSE = sum(residuals.^2, 1);
            SAE = sum(abs(residuals), 1);
            MSE = SSE ./ nSamples;
            RMSE = sqrt(MSE);
            MAE = SAE ./ nSamples;

            centered = bsxfun(@minus, yTrue, mean(yTrue, 1));
            SST = sum(centered.^2, 1);
            R2 = 1 - SSE ./ SST;

            % R2 and FIT are undefined for a constant measured output.
            R2(SST == 0) = NaN;
            FIT = 100 * (1 - sqrt(SSE) ./ sqrt(SST));
            FIT(SST == 0) = NaN;

            switch normalization
                case 'range'
                    scale = max(yTrue, [], 1) - min(yTrue, [], 1);
                case 'std'
                    scale = std(yTrue, 0, 1);
                case 'mean'
                    scale = abs(mean(yTrue, 1));
            end
            NRMSE = RMSE ./ scale;
            NRMSE(scale == 0) = NaN;

            % MAPE ignores samples whose measured value is zero.
            APE = abs(residuals ./ yTrue);
            APE(yTrue == 0) = NaN;
            MAPE = 100 * mltoolbox.metrics.regressionMetrics.meanIgnoringNaN(APE, 1);

            adjustedR2 = NaN(1, nOutputs);
            AIC = NaN(1, nOutputs);
            AICc = NaN(1, nOutputs);
            BIC = NaN(1, nOutputs);
            if ~isempty(nParameters)
                denominator = nSamples - nParameters - 1;
                if denominator > 0
                    adjustedR2 = 1 - (1 - R2) .* ...
                        (nSamples - 1) ./ denominator;
                end

                safeMSE = max(SSE ./ nSamples, realmin);
                AIC = nSamples .* log(safeMSE) + 2 * nParameters;
                BIC = nSamples .* log(safeMSE) + ...
                    nParameters * log(nSamples);
                if nSamples > nParameters + 1
                    AICc = AIC + 2 * nParameters * (nParameters + 1) / ...
                        (nSamples - nParameters - 1);
                end
            end

            metrics.nSamples = nSamples;
            metrics.nOutputs = nOutputs;
            metrics.normalization = normalization;
            metrics.perOutput.SSE = SSE;
            metrics.perOutput.MSE = MSE;
            metrics.perOutput.RMSE = RMSE;
            metrics.perOutput.MAE = MAE;
            metrics.perOutput.NRMSE = NRMSE;
            metrics.perOutput.MAPE = MAPE;
            metrics.perOutput.R2 = R2;
            metrics.perOutput.adjustedR2 = adjustedR2;
            metrics.perOutput.FIT = FIT;
            metrics.perOutput.AIC = AIC;
            metrics.perOutput.AICc = AICc;
            metrics.perOutput.BIC = BIC;

            % Overall errors pool all samples and outputs. R2 and FIT use
            % each output's own mean, avoiding artificial scale offsets.
            totalObservations = nSamples * nOutputs;
            totalSSE = sum(SSE);
            totalSST = sum(SST);
            metrics.overall.SSE = totalSSE;
            metrics.overall.MSE = totalSSE / totalObservations;
            metrics.overall.RMSE = sqrt(metrics.overall.MSE);
            metrics.overall.MAE = sum(SAE) / totalObservations;
            metrics.overall.NRMSE = mltoolbox.metrics.regressionMetrics.meanIgnoringNaN(NRMSE, 2);
            metrics.overall.MAPE = mltoolbox.metrics.regressionMetrics.meanIgnoringNaN(MAPE, 2);
            if totalSST == 0
                metrics.overall.R2 = NaN;
                metrics.overall.FIT = NaN;
            else
                metrics.overall.R2 = 1 - totalSSE / totalSST;
                metrics.overall.FIT = 100 * ...
                    (1 - sqrt(totalSSE) / sqrt(totalSST));
            end
            metrics.overall.adjustedR2 = ...
                mltoolbox.metrics.regressionMetrics.meanIgnoringNaN(adjustedR2, 2);
        end
    end

    methods (Static, Access = private)
        function [yTrue, yPred] = validateSignals(yTrue, yPred)
            validateattributes(yTrue, {'numeric'}, {'2d', 'nonempty', 'real'});
            validateattributes(yPred, {'numeric'}, {'2d', 'nonempty', 'real'});
            if isrow(yTrue), yTrue = yTrue(:); end
            if isrow(yPred), yPred = yPred(:); end
            if ~isequal(size(yTrue), size(yPred))
                error('regressionMetrics:SizeMismatch', ...
                    'yTrue and yPred must have the same size.');
            end
            if any(~isfinite(yTrue(:))) || any(~isfinite(yPred(:)))
                error('regressionMetrics:NonFiniteData', ...
                    'Signals cannot contain NaN or Inf values.');
            end
        end

        function value = meanIgnoringNaN(x, dim)
            mask = ~isnan(x);
            count = sum(mask, dim);
            x(~mask) = 0;
            value = sum(x, dim) ./ count;
            value(count == 0) = NaN;
        end
    end
end
