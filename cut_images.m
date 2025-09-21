% 读取图像
image_path = 'D:\hand_craft_registration\test_SAR\dys-1.tiff';
image = imread(image_path);

% 获取图像的尺寸
[height, width, ~] = size(image);

% 计算切割点
vertical_cut = round(height / 2);
horizontal_cut_1 = round(width / 3);
horizontal_cut_2 = 2 * horizontal_cut_1;

% 切割图像为6份
sub_image_1 = image(1:vertical_cut, 1:horizontal_cut_1, :);
sub_image_2 = image(1:vertical_cut, horizontal_cut_1 + 1:horizontal_cut_2, :);
sub_image_3 = image(1:vertical_cut, horizontal_cut_2 + 1:width, :);
sub_image_4 = image(vertical_cut + 1:height, 1:horizontal_cut_1, :);
sub_image_5 = image(vertical_cut + 1:height, horizontal_cut_1 + 1:horizontal_cut_2, :);
sub_image_6 = image(vertical_cut + 1:height, horizontal_cut_2 + 1:width, :);

% 提取原文件名和扩展名
[pathstr, name, ext] = fileparts(image_path);

% 保存切割后的图像
new_folder = fullfile(pathstr, 'split_dys-1');
if ~exist(new_folder, 'dir')
    mkdir(new_folder);
end

imwrite(sub_image_1, fullfile(new_folder, [name, '_1', '.png']));
imwrite(sub_image_2, fullfile(new_folder, [name, '_2', '.png']));
imwrite(sub_image_3, fullfile(new_folder, [name, '_3', '.png']));
imwrite(sub_image_4, fullfile(new_folder, [name, '_4', '.png']));
imwrite(sub_image_5, fullfile(new_folder, [name, '_5', '.png']));
imwrite(sub_image_6, fullfile(new_folder, [name, '_6', '.png']));

% 显示原始图像和切割后的6份图像
figure;
subplot(3, 3, 1);
imshow(image);
title('Original Image');

subplot(3, 3, 2);
imshow(sub_image_1);
title('Sub-image 1');

subplot(3, 3, 3);
imshow(sub_image_2);
title('Sub-image 2');

subplot(3, 3, 5);
imshow(sub_image_3);
title('Sub-image 3');

subplot(3, 3, 6);
imshow(sub_image_4);
title('Sub-image 4');

subplot(3, 3, 8);
imshow(sub_image_5);
title('Sub-image 5');

subplot(3, 3, 9);
imshow(sub_image_6);
title('Sub-image 6');