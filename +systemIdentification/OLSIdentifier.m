classdef OLSIdentifier < BaseSystemIdentifier

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
            for i=1:length(allNames)
                if ~ismember(allNames{i},{'outputLag','inputLag','errorLag','includeCurrentInput'})
                    regressorPairs = [regressorPairs, allNames{i}, allValues{i}];
                end
            end
            
            % Init internal regressor
            obj.regressor = OLSRegressor(regressorPairs{:});            
            
        end
        
    end
    
end