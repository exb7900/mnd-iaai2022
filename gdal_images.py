import numpy as np
from osgeo import gdal, osr, ogr
from osgeo.gdalconst import *


def read_gdal_image(imPath):
    """
    Read satellite data with GDAL. Based on code from Carl Salvaggio.

    Args:
        imPath: path to data file

    Returns:
        dataset: gdal object / gdal.Open result
        imdata: np array containing image data
        bandNames: list, names of bands in image
        nC: number of columns
        nR: number of rows
        nB: number of bands
        xOrigin: x coordinate (longitude), dataset.GetGeoTransform()[0]
        yOrigin: y coordinate (latitude), dataset.GetGeoTransform()[3]
        pixelWidth: x width, dataset.GetGeoTransform()[1]
        pixelHeight: y height, -dataset.GetGeoTransform()[5]
    """
    dataset = gdal.Open(imPath)
    nC = dataset.RasterXSize
    nR = dataset.RasterYSize
    nB = dataset.RasterCount
    driver = dataset.GetDriver()
    if driver.LongName == 'GeoTIFF':
        geoTiffDataType = \
            gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType)
        if geoTiffDataType.lower() == 'uint8':
            imdata = np.zeros((nR, nC, nB), dtype=np.uint8)
        elif geoTiffDataType.lower() == 'uint16':
            imdata = np.zeros((nR, nC, nB), dtype=np.uint16)
        elif geoTiffDataType.lower() == 'float32':
            imdata = np.zeros((nR, nC, nB), dtype=np.float32)
        elif geoTiffDataType.lower() == 'float64':
            imdata = np.zeros((nR, nC, nB), dtype=np.float32)
        else:
            msg = 'GeoTIFF data type {0} not supported'.format(geoTiffDataType)
            raise TypeError(msg)

    transform = dataset.GetGeoTransform()

    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    bandNames = []
    if nB == 1:
        data = dataset.GetRasterBand(1)
        bandNames.append(data.GetDescription())
        imdata = data.ReadAsArray(0, 0, nC, nR)
    else:
        for band in range(nB):
            data = dataset.GetRasterBand(band+1)
            bandNames.append(data.GetDescription())
            imdata[:,:,band] = data.ReadAsArray(0, 0, nC, nR)

    return dataset, imdata, bandNames, nC, nR, nB, xOrigin, yOrigin, \
        pixelWidth, pixelHeight

def write_gdal_image(imPath, 
                     dataset, 
                     imdata, 
                     bandNames, 
                     nC, 
                     nR, 
                     nB, 
                     xOrigin, 
                     yOrigin, 
                     pixelWidth, 
                     pixelHeight):
    """
    Writes image with GDAL. Based on code from Carl Salvaggio.

    Args:
        imPath: path to write out data file
        dataset: gdal object / gdal.Open result
        imdata: np array containing image data
        bandNames: list, names of bands in image
        nC: number of columns
        nR: number of rows
        nB: number of bands
        xOrigin: x coordinate (longitude), dataset.GetGeoTransform()[0]
        yOrigin: y coordinate (latitude), dataset.GetGeoTransform()[3]
        pixelWidth: x width, dataset.GetGeoTransform()[1]
        pixelHeight: y height, -dataset.GetGeoTransform()[5]
    """
    driver = dataset.GetDriver()
    dIm = driver.Create(imPath, nC, nR, nB, gdal.GDT_Float32)

    dIm.SetGeoTransform((xOrigin,
                        pixelWidth,
                        0,
                        yOrigin, 
                        0,
                        -pixelHeight))
    dIm.SetProjection(dataset.GetProjectionRef())

    if nB == 1:
        dIm.GetRasterBand(1).WriteArray(imdata)
        dIm.GetRasterBand(1).SetDescription(bandNames[0])
    else:
        for band in range(nB):
            dIm.GetRasterBand(band+1).WriteArray(imdata[:,:,band])
            dIm.GetRasterBand(band+1).SetDescription(bandNames[band])
    dIm = None
