classdef shuffle_data
    
    methods (Static)
    
        function [X_shuff, y_shuff] = shuffle(X, y)
            N = size(X,1);
            idx = randperm(N);
            X_shuff = X(idx,:);
            y_shuff = y(idx,:);
        end
        
    end
end