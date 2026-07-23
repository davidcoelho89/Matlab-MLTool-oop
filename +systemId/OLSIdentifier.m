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
            
            addParameter(p,'outputLags',[]);
            addParameter(p,'inputLags',[]);
            addParameter(p,'errorLags',[]);
            addParameter(p,'includeCurrentInput',false);
            addParameter(p,'stepsAhead',1);
            
            
            parse(p,varargin{:});
            idParams = p.Results;
            
            obj.outputLags = idParams.outputLags;
            obj.inputLags = idParams.inputLags;
            obj.errorLags = idParams.errorLags;
            obj.includeCurrentInput = idParams.includeCurrentInput;
            obj.stepsAhead = idParams.stepsAhead;
            
            % Remove sysId hyperparameters
            
            allNames = varargin(1:2:end);
            allValues = varargin(2:2:end);
            
            regressorPairs = {};
            
            idParameterNames = {
                'outputLags', ...
                'inputLags', ...
                'errorLags', ...
                'includeCurrentInput', ...
                'stepsAhead'
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