classdef OLSIdentifier < mltoolbox.baseModels.BaseSystemIdentifier

    % Hyperparameters
    properties
        
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        
    end
    
    methods
        
        function obj = OLSIdentifier(varargin)
            
            % Separate hyperparameters from sysId and the regressor model
            p = inputParser;
            p.KeepUnmatched = true;
            
            addParameter(p,'outputLag',1);
            addParameter(p,'inputLag',0);
            addParameter(p,'errorLag',0);
            addParameter(p,'includeCurrentInput',false);
            
            parse(p,varargin{:});
            idParams = p.Results;
            
            obj.outputLag = idParams.outputLag;
            obj.inputLag = idParams.inputLag;
            obj.errorLag = idParams.errorLag;
            obj.includeCurrentInput = idParams.includeCurrentInput;
            
            % Remove sysId hyperparameters
            
            allNames = varargin(1:2:end);
            allValues = varargin(2:2:end);
            
            regressorPairs = {};
            
            idParameterNames = {
                'outputLag', ...
                'inputLag', ...
                'errorLag', ...
                'includeCurrentInput'
            };
            
            for i=1:length(allNames)
                if ~ismember(allNames{i},idParameterNames)
                    regressorPairs = [regressorPairs, allNames{i}, allValues{i}];
                end
            end
            
            % Init internal regressor
            obj.regressor = mltoolbox.regressors.OLSRegressor(regressorPairs{:});            
            
            obj.modelName = "OLS System Identifier";
            
        end
        
    end
    
end