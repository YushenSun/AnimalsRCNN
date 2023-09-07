from osgeo import gdal

def get_image_size(image_path):
    dataset = gdal.Open(image_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    return width, height

image_path = 'D:/RS/17JUL.tif'  # 替换为您的图像路径
width, height = get_image_size(image_path)

print(f"Image width: {width} pixels")
print(f"Image height: {height} pixels")
