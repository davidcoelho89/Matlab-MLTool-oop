function plot_dataset(X, y, classNames, dims)

if nargin < 3
    classNames = {};
end

[~,nFeatures] = size(X);

% Defines default attributes
if nargin < 4 || isempty(dims)
    if nFeatures >= 3
        dims = 1:3;  % 3 first features
    else
        dims = 1:2;  % 2 first features
    end
end

% Verify if dims is compatible with nFeatures
if any(dims > nFeatures)
    error('Feature index in dims exceeds number of features in X.');
end

classes = unique(y);
nClasses = numel(classes);

% gerar cores distintas
colors = lines(nClasses);

figure; hold on; grid on;

switch length(dims)
    case 2
        view(2)
        for k = 1:nClasses
            idx = y == classes(k);
            scatter(X(idx,dims(1)), X(idx,dims(2)), 50, colors(k,:), 'filled');
        end
        xlabel(sprintf('Feature %d',dims(1))); 
        ylabel(sprintf('Feature %d',dims(2)));
    case 3
        view(3);
        for k = 1:nClasses
            idx = y == classes(k);
            scatter3(X(idx,dims(1)), ...
                     X(idx,dims(2)), ...
                     X(idx,dims(3)), ...
                     50, colors(k,:), 'filled');
        end
        xlabel(sprintf('Feature %d',dims(1))); 
        ylabel(sprintf('Feature %d',dims(2)));
        zlabel(sprintf('Feature %d',dims(3)));
    otherwise
        error('dims must have length 2 (2D) or 3 (3D).');
end

if ~isempty(classNames)
    legend(classNames,'Location','best');
else
    legend(arrayfun(@num2str,classes,'UniformOutput',false),'Location','best');
end

title('Dataset Visualization');
hold off;

end