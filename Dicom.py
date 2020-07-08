import pydicom as dicom
import numpy as np

# Linear transformation : convert to Hounsfield units (HU)
def getHuPixels(pixels_array, rows, cols, intercept, slope):
  m = np.zeros((rows, cols), np.int16)
  for i in range(rows):
    for j in range(cols):
      m[i][j] = pixels_array[i][j] * slope + intercept
  return m

def linearTransform(x, min, max, a, b):
  return (b - a) * ((x - min) / (max - min)) + a

def getSegmentedPixelColor(value, intervals):
  for i in range(len(intervals)):
    if value >= intervals[i][0] and value <= intervals[i][1]:
      b = linearTransform(
          value, intervals[i][0], intervals[i][1], 0.5 * intervals[i][2][0], intervals[i][2][0])
      g = linearTransform(
          value, intervals[i][0], intervals[i][1], 0.5 * intervals[i][2][1], intervals[i][2][1])
      r = linearTransform(
          value, intervals[i][0], intervals[i][1], 0.5 * intervals[i][2][2], intervals[i][2][2])
      return [b, g, r]
  return [0, 0, 0]

class Dicom:
  def __init__(self, src, rescale=True):
    ds = dicom.dcmread(src)
    self.__rows = ds.Rows
    self.__cols = ds.Columns
    self.__patientId = ds.PatientID
    if rescale:
      self.__pixelArray = getHuPixels(
          ds.pixel_array, self.__rows, self.__cols, ds.RescaleIntercept, ds.RescaleSlope)
    else:
      self.__pixelArray = ds.pixel_array

  def getPixelsArray(self):
    return self.__pixelArray

  def getSegmentedBGR(self, intervals):
    image = np.zeros((self.__rows, self.__cols, 3), np.uint8)
    for i in range(self.__rows):
      for j in range(self.__cols):
        image[i][j] = getSegmentedPixelColor(
            self.__pixelArray[i][j], intervals)
    return image
