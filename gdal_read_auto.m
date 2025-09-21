function [img_data, img_width, img_height, img_geotransform, img_projection] = gdal_read_auto(img_file_path, need_data)
% GDAL_READ_AUTO 读取GeoTIFF影像数据及地理信息
% 输入参数：
%   img_file_path - 影像文件路径
%   need_data     - 是否读取像素数据（默认true）
% 输出参数：
%   img_data        - 影像数据矩阵（HxWxC）
%   img_width       - 影像宽度（像素）
%   img_height      - 影像高度（像素）
%   img_geotransform - 6参数仿射变换矩阵[2,4](@ref)
%   img_projection   - 投影信息字符串（WKT格式）

if nargin < 2
    need_data = true;
end

% 读取地理信息
try
    info = geotiffinfo(img_file_path);
    img_width = info.Width;
    img_height = info.Height;
    
    % 构建仿射变换参数[2,4](@ref)
    img_geotransform = [...
        info.GeoTIFFTags.ModelTiepointTag(4),...  % 左上角X坐标
        info.GeoTIFFTags.ModelPixelScaleTag(1),...% X方向分辨率
        0,...                                      % 旋转参数（通常为0）
        info.GeoTIFFTags.ModelTiepointTag(5),...  % 左上角Y坐标
        0,...                                      % 旋转参数（通常为0）
        -info.GeoTIFFTags.ModelPixelScaleTag(2)...% Y方向分辨率（取负值）
    ];
    
    % 获取WKT格式投影信息[9](@ref)
    img_projection = info.GeoTIFFTags.GTModelTypeGeoKey;
    
catch ME
    error('文件读取失败: %s', ME.message);
end

% 读取像素数据
if need_data
    try
        [img_data, ~] = geotiffread(img_file_path);
        % 调整维度顺序为HxWxC[7](@ref)
        if ndims(img_data) == 3
            img_data = permute(img_data, [2 1 3]);
        else
            img_data = permute(img_data, [2 1]);
        end
    catch
        warning('像素数据读取失败，返回空矩阵');
        img_data = [];
    end
else
    img_data = [];
end
end