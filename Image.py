import cv2 as cv
import numpy as np

class Image:
  def __init__(self, pixelsData, name):
    self.__img = pixelsData
    self.__name = name

  def morphOperations(self, kernelSize, morphType):
    # Open and close morph operations
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    if morphType == 'OPEN':
      self.__img = cv.morphologyEx(self.__img, cv.MORPH_OPEN, kernel)
    elif morphType == 'CLOSE':
      self.__img = cv.morphologyEx(self.__img, cv.MORPH_CLOSE, kernel)

  def BGR2HSV(self):
    # Convert to HSV
    self.__img = cv.cvtColor(self.__img, cv.COLOR_BGR2HSV)

  def filterByHSV(self, lowerHSV, higherHSV):
    # InRange filter
    self.__img = cv.inRange(self.__img, lowerHSV, higherHSV)

  def show(self):
    cv.imshow(self.__name, self.__img)
    cv.waitKey(0)
