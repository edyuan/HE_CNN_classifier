mkdir(fullfile('Slide7','Temp','result_fig'));

files = dir(fullfile('Slide7','Temp','1CK_registered','*.png'));

for i = 1:length(files)
    fish = imread(fullfile('Slide7','Temp','1FISH_registered',files(i).name));
    cell_seg = imread(fullfile('Slide7','Temp','7Cell_separation_result',files(i).name));
    BW = boundarymask(cell_seg);
    rgb = imoverlay(fish,BW,'cyan');
    T = readtable(fullfile('Slide7','Temp','9Spreadsheet',[files(i).name(1:end-4),'.csv']));
    for j = 1:length(T.label)
        rgb = insertText(rgb,[T.y(j)+5,T.x(j)+5],num2str(T.totalGreen(j)),'TextColor','green','FontSize',8,'BoxOpacity',0);
        rgb = insertText(rgb,[T.y(j)-5,T.x(j)-5],num2str(T.totalRed(j)),'TextColor','red','FontSize',8,'BoxOpacity',0);
    end
    imwrite(rgb,fullfile('Slide7','Temp','result_fig',files(i).name));
end