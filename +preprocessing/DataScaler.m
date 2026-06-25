classdef DataScaler < handle
    
    % Hyperparameters
    properties
        mode = "zscore";   % "zscore" | "minmax" | "bipolar"
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        minX
        maxX
        meanX
        stdX
    end

    methods
        
        % Constructor
        function obj = DataScaler(varargin)
            
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

        function obj = fit(obj, X)
            obj.meanX = mean(X,1);
            obj.stdX = std(X,0,1);
            obj.minX = min(X,[],1);
            obj.maxX = max(X,[],1);
        end

        function Xnorm = transform(obj, X)
            switch obj.mode
                case "zscore"
                    Xnorm = (X - obj.meanX) ./ obj.stdX;
                case "minmax"
                    Xnorm = (X - obj.minX) ./ (obj.maxX - obj.minX);
                case "bipolar"
                    Xnorm = 2*(X - obj.minX) ./ (obj.maxX - obj.minX) - 1;
            end
        end

        function Xnorm = fit_transform(obj, X)
            obj.fit(X);
            Xnorm = obj.transform(X);
        end
    end
end