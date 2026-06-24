classdef LabelEncoder < handle

    % Hyperparameters
    properties
        mode = "integer";   % "integer" | "onehot" | "bipolar"
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        classLabels = [];
        nClasses = 0
    end
    
    methods

        function obj = LabelEncoder(varargin)
            
            if mod(nargin,2) ~= 0
                error('Arguments must be given as name-value pairs.');
            end
            
            for k = 1:2:length(varargin)
                name = varargin{k};
                value = varargin{k+1};

                if ~isprop(obj, name)
                    error('Unknown property: %s', string(name));
                end

                obj.(name) = value;
            end
        end

        function obj = fit(obj, y)

            obj.classLabels = unique(y, 'stable');
            obj.nClasses = numel(obj.classLabels);

        end

        function Y = transform(obj, y)

            obj.check_fitted();

            switch obj.mode

                case "integer"
                    Y = obj.toInteger(y);

                case "onehot"
                    Y = obj.toOneHot(y);

                case "bipolar"
                    Y = obj.toBipolar(y);

                otherwise
                    error("Unknown encoding mode.");
            end
        end

        function Y = fit_transform(obj, y)
            obj.fit(y);
            Y = obj.transform(y);
        end

        function y = inverse_transform(obj, Y)

            obj.check_fitted();

            if isvector(Y) && size(Y,2) == 1
                idx = round(Y);
                idx = max(1, min(idx, obj.nClasses));
                y = obj.classLabels(idx);
                return;
            end

            % for matrix outputs (onehot / bipolar / scores)
            [~, idx] = max(Y, [], 2);
            y = obj.classLabels(idx);

        end

    end

    methods (Access = private)

        function Y = toInteger(obj, y)
             [~, Y] = ismember(y, obj.classLabels);
        end

        function Y = toOneHot(obj, y)

            N = numel(y);
            Y = zeros(N, obj.nClasses);

            for k = 1:obj.nClasses
                Y(:,k) = (y == obj.classLabels(k));
            end
        end

        function Y = toBipolar(obj, y)

            N = numel(y);
            Y = -ones(N, obj.nClasses);

            for k = 1:obj.nClasses
                Y(:,k) = 2*(y == obj.classLabels(k)) - 1;
            end
        end

        function check_fitted(obj)
            if isempty(obj.classLabels)
                error("LabelEncoder not fitted.");
            end
        end

    end
end