classdef IrisDataset

    properties
        X
        y
        classes
    end

    methods
    
        function obj = IrisDataset()
            % Construtor: carrega os dados ao instanciar
            [obj.X, obj.y, obj.classes] = obj.load();
        end

        function [X, y, classes] = load(~)
            % Define caminho do arquivo local dentro do package +datasets
            thisFilePath = mfilename('fullpath');
            thisFolder = fileparts(thisFilePath);
            fname = fullfile(thisFolder,'iris.txt');

            % Lê o arquivo
            T = readtable(fname,'ReadVariableNames',false);

            % Remove linhas vazias
            T = T(~all(ismissing(T),2),:);

            % Extrai atributos e labels
            X = table2array(T(:,1:4));
            y_str = table2array(T(:,5));

            % Converte labels em integers
            classes = unique(y_str,'stable');
            y = zeros(size(y_str));

            for k = 1:numel(classes)
                y(strcmp(y_str,classes{k})) = k;
            end
        end
    end
end