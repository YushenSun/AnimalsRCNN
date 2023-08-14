from osgeo import gdal

# Path to the input 8-band TIFF file
input_tiff_path = "D:/RS/17JUL.tif"

# Open the input TIFF file
dataset = gdal.Open(input_tiff_path)

if dataset is None:
    print(f"Error: Could not open input TIFF file '{input_tiff_path}'")
    exit(1)

# Get the number of rows, columns, and bands in the dataset
rows = dataset.RasterYSize
cols = dataset.RasterXSize
num_bands = dataset.RasterCount

# Specify the bands to use for each color channel
blue_bands = [1, 2, 3]  # Bands for blue channel (averaged)
green_bands = [4, 5, 6]  # Bands for green channel (averaged)
red_bands = [7, 8]  # Bands for red channel (averaged)

# Read the pixel values from the specified bands and calculate averages
blue_data = sum(dataset.GetRasterBand(band).ReadAsArray() for band in blue_bands) / len(blue_bands)
green_data = sum(dataset.GetRasterBand(band).ReadAsArray() for band in green_bands) / len(green_bands)
red_data = sum(dataset.GetRasterBand(band).ReadAsArray() for band in red_bands) / len(red_bands)

# Create a new RGB image using the averaged data
rgb_array = [red_data, green_data, blue_data]

# Convert the RGB data to an 8-bit unsigned integer
rgb_array = [((array - array.min()) / (array.max() - array.min()) * 255).astype('uint8') for array in rgb_array]

# Create a new dataset to save the RGB image
output_tiff_path = "D:/RS/17JULRGB.tif"
driver = gdal.GetDriverByName('GTiff')

output_dataset = driver.Create(output_tiff_path, cols, rows, 3, gdal.GDT_Byte)

if output_dataset is None:
    print(f"Error: Could not create output TIFF file '{output_tiff_path}'")
    exit(1)

# Write the RGB data to the new dataset
for i, band_data in enumerate(rgb_array, start=1):
    output_band = output_dataset.GetRasterBand(i)
    if output_band is None:
        print(f"Error: Could not access band {i} of the output dataset")
        exit(1)

    output_band.WriteArray(band_data)
    output_band.FlushCache()

# Set the projection and geotransform from the input dataset
output_dataset.SetProjection(dataset.GetProjection())
output_dataset.SetGeoTransform(dataset.GetGeoTransform())

# Close the datasets
dataset = None
output_dataset = None

print("RGB image saved:", output_tiff_path)
