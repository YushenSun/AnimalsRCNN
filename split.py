from osgeo import gdal
import os

def split_image(input_image_path, output_folder, block_size, overlap):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dataset = gdal.Open(input_image_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    m=0
    n=0

    for i in range(0, width, block_size - overlap):
        n=0
        for j in range(0, height, block_size - overlap):
            x_offset = i
            y_offset = j
            x_size = min(block_size, width - i)
            y_size = min(block_size, height - j)


            output_filename = os.path.join(output_folder, f"block_{m}_{n}.tif")
            n = n + 1
            gdal.Translate(output_filename, dataset, format="GTiff", srcWin=(x_offset, y_offset, x_size, y_size))

        m=m+1
    dataset = None

if __name__ == "__main__":
    '''
    input_image_path = "D:/RS/17JULRGB_linear.tif"
    output_folder = "D:/RS/Blocks_17JULRGB_linear"
    block_size = 1024
    overlap = 10

    split_image(input_image_path, output_folder, block_size, overlap)
    '''
    input_image_path = "D:/RS/Blocks_17JULRGB_linear/block_0_0.tif"
    output_folder = "D:/RS/Blocks_17JULRGB_linear_small"
    block_size = 16
    overlap = 0

    split_image(input_image_path, output_folder, block_size, overlap)
