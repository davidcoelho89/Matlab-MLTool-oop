classdef OLSClassifier < mltoolbox.classifiers.LinearClassifier
    % 
    % --- OLSCLASSIFIER - Ordinary Least-Squares Classifier ---
    %
    % Library Convetion:
    %   X : [N x p]
    %   y : [N x 1]
    %
    % Properties (Hyperparameters)
    %
    %   approximation: which aproximation method is used
    %     'pinv'       : W = Y*pinv(X);
    %     'svd'        : W = Y/X;
    %     'theoretical': W = Y*X'/(X*X' + regularization * eye(p,p));
    %   regularization: used to mitigate numerical computation errors
    %     (constant)
    %
    % Properties (Parameters)
    %
    %   .
    %   
    % Methods (for external use)
    %
    %   .
    %
    % Methods (protected)
    %
    %   .
    %
    % ----------------------------------------------------------------

    % Hyperparameters
    properties
        approximation = 'pinv';     % | 'svd' | 'theoretical' |
        regularization = 0.0001;
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        
    end
    
    methods
        
        % Constructor
        function obj = OLSClassifier(varargin)
            
            obj.modelName = "OLS Classifier";
            
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
        
        % Training Function (N instances)
        function obj = fit(obj,X,y)
            
            X = obj.addBiasTerm(X);
            
            obj.validateFitInputs(X,y)
            
            if ~isempty(obj.encoder)
                Y = obj.encoder.transform(y);
            else
                [Y, classLabels] = obj.oneHotEncodeLabels(y);
                obj.classLabels = classLabels;
            end
            
            obj.nClasses = size(Y,2);
            obj.nFeatures = size(X,2);
            
            switch obj.approximation
                case 'pinv'
                    obj.W = pinv(X) * Y;
                 case 'svd'
                    obj.W = X \ Y;
                case 'theoretical'
                    p = size(X,2);
                    obj.W = (X'*X + obj.regularization*eye(p)) \ (X'*Y);
                otherwise
                    obj.W = pinv(X) * Y;                    
            end
            
            obj.isTrained = true;
            
        end
        
    end % end methods
    
end % end class