# Our imports
from Dicom import Dicom
from os import listdir
from os.path import isfile, join
from Image import Image

# Initialize instances data
craniumBySide = Image(Image.read(
    './bonePartsInstances/cranium.png', 'gray'), 'cranium')
rightLeg = Image(Image.read(
    './bonePartsInstances/rightLeg.png', 'gray'), 'rightLeg')


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

  boneImg.bgr2hsv()
  boneImg.filterByHSV((100, 50, 0), (130, 255, 255))  # Image becomes gray!

  metastasisImg.bgr2hsv()
  metastasisImg.filterByHSV((0, 60, 0), (10, 255, 255))  # Image becomes gray!

  # Perform some morph operations
  boneImg.morphOperations(2, 'OPEN')
  boneImg.morphOperations(2, 'CLOSE')

  metastasisImg.morphOperations(2, 'OPEN')
  metastasisImg.morphOperations(10, 'CLOSE')

  instancePts = boneImg.findInstance(craniumBySide, 5)
  boneImg.gray2bgr()
  if instancePts != None:
    boneImg.drawPolylines(instancePts)
  else:
    print('No instance found :(')

  metastasisImg.findCountours(1)
  features = metastasisImg.findContoursFeatures()

  # Here we split the bone image into bone parts, so we can discover witch one of them is suffering from metastasis
  # In this code, we are using instance detection using SIFT and homography or template matching by Normalized cross correlation

  metastasisImg.gray2bgr()
  for i in range(0, len(features)):  # Iterate over detected metastasis
    # Draw centroid and approximate convex hull for each one of them
    metastasisImg.drawCircle(features[i]['centroid'])
    metastasisImg.drawContours([features[i]['convexHull']])

  # boneImg.show()
  # metastasisImg.show()
