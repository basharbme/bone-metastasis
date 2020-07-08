# Our imports
from Dicom import Dicom
from os import listdir
from os.path import isfile, join
from Image import Image


# Here it comes!
path = "./datasets/"
dcmFiles = []
dirFiles = listdir(path)
i = 0
while i < len(dirFiles):
  if isfile(join(path, dirFiles[i])) and dirFiles[i].find(".dcm"):
    dcmFiles.append(dirFiles[i])
  i = i + 1

for filename in dcmFiles:

  ds = Dicom(path+filename, False)

  # Filter by Hounsfield units (HU)

  segmentedBGR = ds.getSegmentedBGR([
      [7, 100, (255, 0, 0)], [50, 200, (0, 0, 255)]])

  boneImg = Image(segmentedBGR, "Bone")
  metastasisImg = Image(segmentedBGR, "Metastasis")

  boneImg.BGR2HSV()
  boneImg.filterByHSV((100, 50, 0), (130, 255, 255))

  metastasisImg.BGR2HSV()
  metastasisImg.filterByHSV((0, 60, 0), (10, 255, 255))

  # Perform some morph operations
  boneImg.morphOperations(2, 'OPEN')
  boneImg.morphOperations(2, 'CLOSE')

  metastasisImg.morphOperations(2, 'OPEN')
  metastasisImg.morphOperations(10, 'CLOSE')

  boneImg.show()
  metastasisImg.show()
