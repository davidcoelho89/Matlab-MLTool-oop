classdef train_test_split
    
    methods (Static)
        
        function [Xtr,Xts,ytr,yts] = split(X,y,varargin)
            p = inputParser;
            addParameter(p,'train_ratio',0.7,@(x) x>0 & x<1);
            addParameter(p,'shuffle',true,@islogical);
            parse(p,varargin{:});
            tr_ratio = p.Results.train_ratio;
            shuffle = p.Results.shuffle;

            if shuffle
                [X,y] = mltoolbox.preprocessing.shuffle_data.shuffle(X,y);
            end

            N = size(X,1);
            Ntr = round(tr_ratio*N);
            train_idx = 1:Ntr;
            test_idx  = Ntr+1:N;

            Xtr = X(train_idx,:);
            Xts = X(test_idx,:);
            ytr = y(train_idx,:);
            yts = y(test_idx,:);
        end
        
    end
    
end