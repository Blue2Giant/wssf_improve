img_file_path = "D:\registration_HT\Clipped.tif";
img_geo_transform = imfinfo(img_file_path);
disp(img_geotransform.ModelPixelScaleTag[:2])
disp(img_geo_transform)
imshow(img_data)