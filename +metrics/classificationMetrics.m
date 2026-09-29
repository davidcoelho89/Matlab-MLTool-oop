classdef classificationMetrics
    %CLASSIFICATIONMETRICS Metrics for hard multiclass classification.
    %
    %   metrics = classificationMetrics.calculate(YTrue, YPred)
    %
    %   YTrue and YPred must both be N-by-Nc one-hot matrices. Each row
    %   must contain exactly one 1 and zeros elsewhere. Rows are samples;
    %   columns are classes. YPred represents hard class decisions, not
    %   continuous class scores.
    %
    %   Optional name-value argument:
    %       'ClassLabels' 1-by-Nc or Nc-by-1 class names/labels.
    %
    %   The confusion matrix convention is rows = actual class and
    %   columns = predicted class.

    methods (Static)
        function metrics = calculate(YTrue, YPred, varargin)
            p = inputParser;
            addParameter(p, 'ClassLabels', []);
            parse(p, varargin{:});

            classLabels = p.Results.ClassLabels;
            [YTrue, YPred, nSamples, nClasses] = ...
                mltoolbox.metrics.classificationMetrics.validateOneHot( ...
                YTrue, YPred);

            if isempty(classLabels)
                classLabels = (1:nClasses)';
            else
                if ~isvector(classLabels) || numel(classLabels) ~= nClasses
                    error('classificationMetrics:InvalidClassLabels', ...
                        'ClassLabels must contain one label per one-hot column.');
                end
                classLabels = classLabels(:);
            end

            [~, trueClass] = max(YTrue, [], 2);
            [~, predictedClass] = max(YPred, [], 2);

            % Rows are true classes; columns are predicted classes.
            confusionMatrix = accumarray( ...
                [trueClass, predictedClass], 1, [nClasses, nClasses]);

            TP = diag(confusionMatrix);
            support = sum(confusionMatrix, 2);
            predictedCount = sum(confusionMatrix, 1)';
            FP = predictedCount - TP;
            FN = support - TP;
            TN = nSamples - TP - FP - FN;

            precision = mltoolbox.metrics.classificationMetrics.safeDivide( ...
                TP, TP + FP);
            recall = mltoolbox.metrics.classificationMetrics.safeDivide( ...
                TP, TP + FN);
            specificity = mltoolbox.metrics.classificationMetrics.safeDivide( ...
                TN, TN + FP);
            fpr = mltoolbox.metrics.classificationMetrics.safeDivide( ...
                FP, FP + TN);
            fnr = mltoolbox.metrics.classificationMetrics.safeDivide( ...
                FN, FN + TP);
            f1 = mltoolbox.metrics.classificationMetrics.safeDivide( ...
                2 .* precision .* recall, precision + recall);
            mccPerClass = mltoolbox.metrics.classificationMetrics.binaryMCC( ...
                TP, FP, TN, FN);

            correct = sum(TP);
            accuracy = correct / nSamples;

            % Micro metrics aggregate the one-vs-rest counts.
            microTP = sum(TP);
            microFP = sum(FP);
            microFN = sum(FN);
            microPrecision = mltoolbox.metrics.classificationMetrics.safeDivide( ...
                microTP, microTP + microFP);
            microRecall = mltoolbox.metrics.classificationMetrics.safeDivide( ...
                microTP, microTP + microFN);
            microF1 = mltoolbox.metrics.classificationMetrics.safeDivide( ...
                2 * microPrecision * microRecall, ...
                microPrecision + microRecall);

            metrics.nSamples = nSamples;
            metrics.nClasses = nClasses;
            metrics.classLabels = classLabels;
            metrics.confusionMatrix = confusionMatrix;
            metrics.accuracy = accuracy;
            metrics.errorRate = 1 - accuracy;

            metrics.perClass.TP = TP;
            metrics.perClass.FP = FP;
            metrics.perClass.TN = TN;
            metrics.perClass.FN = FN;
            metrics.perClass.support = support;
            metrics.perClass.precision = precision;
            metrics.perClass.recall = recall;
            metrics.perClass.sensitivity = recall;
            metrics.perClass.specificity = specificity;
            metrics.perClass.falsePositiveRate = fpr;
            metrics.perClass.falseNegativeRate = fnr;
            metrics.perClass.F1 = f1;
            metrics.perClass.MCC = mccPerClass;

            metrics.macro.precision = mean(precision, 'omitnan');
            metrics.macro.recall = mean(recall, 'omitnan');
            metrics.macro.specificity = mean(specificity, 'omitnan');
            metrics.macro.F1 = mean(f1, 'omitnan');
            metrics.macro.MCC = mean(mccPerClass, 'omitnan');
            metrics.balancedAccuracy = metrics.macro.recall;

            weights = support / nSamples;
            metrics.weighted.precision = sum(weights .* precision, 'omitnan');
            metrics.weighted.recall = sum(weights .* recall, 'omitnan');
            metrics.weighted.specificity = sum(weights .* specificity, 'omitnan');
            metrics.weighted.F1 = sum(weights .* f1, 'omitnan');
            metrics.weighted.MCC = sum(weights .* mccPerClass, 'omitnan');

            metrics.micro.precision = microPrecision;
            metrics.micro.recall = microRecall;
            metrics.micro.F1 = microF1;

            % Generalized multiclass Matthews correlation coefficient.
            rowTotals = sum(confusionMatrix, 2);
            columnTotals = sum(confusionMatrix, 1)';
            numerator = correct * nSamples - sum(rowTotals .* columnTotals);
            denominator = sqrt((nSamples^2 - sum(columnTotals.^2)) * ...
                (nSamples^2 - sum(rowTotals.^2)));
            metrics.MCC = mltoolbox.metrics.classificationMetrics.safeDivide( ...
                numerator, denominator);

            % ROC/AUC are intentionally not reported: hard one-hot
            % predictions do not provide continuous class scores.
        end
    end

    methods (Static, Access = private)
        function [YTrue, YPred, nSamples, nClasses] = validateOneHot( ...
                YTrue, YPred)
            validateattributes(YTrue, {'numeric','logical'}, ...
                {'2d','nonempty','real'}, mfilename, 'YTrue');
            validateattributes(YPred, {'numeric','logical'}, ...
                {'2d','nonempty','real'}, mfilename, 'YPred');

            if ~isequal(size(YTrue), size(YPred))
                error('classificationMetrics:SizeMismatch', ...
                    'YTrue and YPred must have identical N-by-Nc dimensions.');
            end
            if any(~isfinite(YTrue(:))) || any(~isfinite(YPred(:)))
                error('classificationMetrics:NonFiniteData', ...
                    'YTrue and YPred cannot contain NaN or Inf.');
            end
            if any((YTrue(:) ~= 0) & (YTrue(:) ~= 1)) || ...
                    any((YPred(:) ~= 0) & (YPred(:) ~= 1))
                error('classificationMetrics:NotOneHot', ...
                    'YTrue and YPred must contain only zeros and ones.');
            end

            nSamples = size(YTrue, 1);
            nClasses = size(YTrue, 2);
            if nClasses < 2
                error('classificationMetrics:TooFewClasses', ...
                    'At least two class columns are required.');
            end
            if any(sum(YTrue, 2) ~= 1) || any(sum(YPred, 2) ~= 1)
                error('classificationMetrics:InvalidOneHotRows', ...
                    'Every row must contain exactly one active class.');
            end
        end

        function q = safeDivide(a, b)
            q = NaN(size(a + b));
            valid = (b ~= 0);
            q(valid) = a(valid) ./ b(valid);
        end

        function value = binaryMCC(TP, FP, TN, FN)
            numerator = TP .* TN - FP .* FN;
            denominator = sqrt((TP + FP) .* (TP + FN) .* ...
                (TN + FP) .* (TN + FN));
            value = mltoolbox.metrics.classificationMetrics.safeDivide( ...
                numerator, denominator);
        end
    end
end
