% Kernel density estimation of the coupling coordination degree (CCD).
%
% Reads output/ccd_panel.csv, written by main_analysis.py, and produces one 3D
% density surface per region (overall, Ganpo, Middle Reaches, Downstream) for
% 2010-2023.  The bandwidth (0.04 on the CCD scale) matches the manuscript.
%
% Requirements: MATLAB R2020a or later (readtable, exportgraphics).
% Run from the package root:  matlab -batch "run('code/kernel_density.m')"

clear; clc;

baseDir = fileparts(fileparts(mfilename('fullpath')));
inFile = fullfile(baseDir, 'output', 'ccd_panel.csv');
outDir = fullfile(baseDir, 'figures');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
if ~isfile(inFile)
    error('Input file not found: %s. Run main_analysis.py first.', inFile);
end

panel = readtable(inFile, 'TextType', 'string');
panel.Year = double(panel.Year);
panel.D = double(panel.D);
if ~ismember('Region', panel.Properties.VariableNames)
    error('The input file must contain a Region column.');
end

years = 2010:2023;
grid = linspace(0, 1, 800);
bandwidth = 0.04;

regions = ["All", "Ganpo", "Middle Reaches", "Downstream"];
tags = ["overall", "ganpo", "middle", "downstream"];

for k = 1:numel(regions)
    if regions(k) == "All"
        subset = panel;
    else
        subset = panel(panel.Region == regions(k), :);
    end

    density = nan(numel(years), numel(grid));
    for i = 1:numel(years)
        values = subset.D(subset.Year == years(i));
        if isempty(values)
            continue;
        end
        [f, ~] = ksdensity(values, grid, 'Bandwidth', bandwidth);
        density(i, :) = f;
    end

    valid = ~all(isnan(density), 2);
    if ~any(valid)
        warning('No data for region %s; skipped.', regions(k));
        continue;
    end

    [X, Y] = meshgrid(grid, years);
    fig = figure('Position', [100, 100, 1000, 800], ...
                 'Name', "KDE of CCD - " + regions(k));
    mesh(X(valid, :), Y(valid, :), density(valid, :), ...
         'EdgeColor', 'interp', 'FaceAlpha', 0.6);
    colormap(gca, cividis_like(512));
    colorbar;

    xlabel('Coupling Coordination Degree', 'FontSize', 12, ...
           'FontWeight', 'bold', 'FontUnits', 'points');
    ylabel('Year', 'FontSize', 12, 'FontWeight', 'bold', ...
           'FontUnits', 'points');
    zlabel('Kernel density estimates', 'FontSize', 12, ...
           'FontWeight', 'bold', 'FontUnits', 'points');
    title("3D Kernel Density Estimation of Coupling Coordination (" + ...
          regions(k) + ", 2010-2023)", 'FontSize', 16, ...
          'FontWeight', 'bold', 'Interpreter', 'none', ...
          'FontUnits', 'points');

    xlim([0.3 1]);
    ylim([min(years) max(years)]);
    zmax = max(density(:), [], 'omitnan');
    if isempty(zmax) || isnan(zmax) || zmax == 0
        zmax = 8;
    else
        zmax = ceil(zmax * 1.05);
        if zmax <= 0
            zmax = 1;
        end
    end
    zlim([0 zmax]);

    pbaspect([2.5 2 1]);
    view([-35, 30]);
    grid on;
    set(gca, 'GridLineStyle', '-', 'FontSize', 10, ...
        'GridColor', [0.3 0.3 0.3], 'GridAlpha', 0.5, ...
        'FontName', 'Times New Roman');
    set(fig, 'Color', 'white');
    lighting phong;
    material shiny;
    xticks(0.3:0.1:1);
    yticks(years);
    if zmax <= 10
        ztick_step = 1;
    else
        ztick_step = ceil(zmax / 5);
    end
    zticks(0:ztick_step:zmax);

    pdfFile = fullfile(outDir, "kde_" + tags(k) + ".pdf");
    pngFile = fullfile(outDir, "kde_" + tags(k) + ".png");
    try
        exportgraphics(fig, pdfFile, 'ContentType', 'vector');
        exportgraphics(fig, pngFile, 'Resolution', 600);
    catch
        print(fig, pdfFile, '-dpdf', '-bestfit');
        print(fig, pngFile, '-dpng', '-r600');
    end
    close(fig);
    fprintf('Wrote %s\n', pdfFile);
end

function cmap = cividis_like(n)
    % A cividis-like map used for the density surfaces in the manuscript.
    half = n / 2;
    r = [linspace(0.0, 0.5, half), linspace(0.5, 1.0, half)];
    g = [linspace(0.0, 0.7, half), linspace(0.7, 1.0, half)];
    b = [linspace(0.5, 0.9, half), linspace(0.9, 0.0, half)];
    cmap = [r', g', b'];
    cmap(1, :) = [0 0 0];
end
