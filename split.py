from osgeo import gdal
import os

def split_image(input_image_path, output_folder, block_size, overlap):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dataset = gdal.Open(input_image_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    for i in range(0, width, block_size - overlap):
        for j in range(0, height, block_size - overlap):
            x_offset = i
            y_offset = j
            x_size = min(block_size, width - i)
            y_size = min(block_size, height - j)

            output_filename = os.path.join(output_folder, f"block_{i}_{j}.tif")

            gdal.Translate(output_filename, dataset, format="GTiff", srcWin=(x_offset, y_offset, x_size, y_size))

    dataset = None

if __name__ == "__main__":
    input_image_path = "D:/RS/20SEP.tif"
    output_folder = "D:/RS/Blocks"
    block_size = 1024
    overlap = 10

    split_image(input_image_path, output_folder, block_size, overlap)
