classdef OLSClassifier < mltoolbox.classifiers.linearClassifier
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
        approximation = 'pinv';
        regularization = 0.0001;
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        modelName = 'ols';
    end
    
    methods
        
        % Constructor
        function self = OLSClassifier()
            % Set the hyperparameters after initializing!
        end
        
        % Training Function (N instances)
        function self = fit(self,X,Y)
            
            [p,N] = size(X);
            if(self.add_bias)
                p = p+1;
                X = [ones(1,N) ; X];
            end
            
            if(strcmp(self.approximation,'pinv'))
                self.W = Y*pinv(X);
            elseif(strcmp(self.approximation,'svd'))
                self.W = Y/X;
            elseif(strcmp(self.approximation,'theoretical'))
                self.W = Y*X'/(X*X' + self.regularization * eye(p,p));
            else
                self.W = Y*pinv(X);
            end            
            
        end
        
    end % end methods
    
end % end class