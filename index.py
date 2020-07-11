# Our imports
from Dicom import Dicom
from os import listdir
from os.path import isfile, join
from Image import Image
from geometry import isPointsInsidePolygon

# Initialize instances data
craniumBySide = Image(Image.read(
    './bonePartsInstances/cranium.png', 'gray'), 'cranium')
leg = Image(Image.read(
    './bonePartsInstances/rightLeg.png', 'gray'), 'leg')
chest = Image(Image.read(
    './bonePartsInstances/chest.png', 'gray'), 'chest')
arm = Image(Image.read(
    './bonePartsInstances/rightArm.png', 'gray'), 'arm')
waist = Image(Image.read(
    './bonePartsInstances/waist.png', 'gray'), 'waist')


# Here it comes!
path = "./datasets/"
dcmFiles = []
dirFiles = listdir(path)
i = 0
while i < len(dirFiles):
  if isfile(join(path, dirFiles[i])) and dirFiles[i].find(".dcm"):
    dcmFiles.append(dirFiles[i])
  i = i + 1

results = open("./results.txt", "a")

for filename in dcmFiles:

  ds = Dicom(path+filename, False)
  results.write(str(ds.getPatientId()) + ':\n')
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

  craniumBySideRect = boneImg.templateMatch(craniumBySide)
  legRect = boneImg.templateMatch(leg)

  metastasisImg.findCountours(1)
  features = metastasisImg.findContoursFeatures()

  availableBoneParts = []
  if len(features) > 0:
    # Here we split the bone image into bone parts, so we can discover witch one of them is suffering from metastasis
    # In this code, we are using instance detection using SIFT and homography or template matching by Normalized cross correlation

    craniumBySideRect = boneImg.templateMatch(craniumBySide)
    if craniumBySideRect != None:
      availableBoneParts.append(
          dict(name=craniumBySide.getName(), polygon=craniumBySideRect))

    legRect = boneImg.templateMatch(leg)
    if legRect != None:
      availableBoneParts.append(
          dict(name=leg.getName(), polygon=legRect))

    armRect = boneImg.templateMatch(arm)
    if armRect != None:
      availableBoneParts.append(
          dict(name=arm.getName(), polygon=armRect))

    chestRect = boneImg.templateMatch(chest)
    if chestRect != None:
      availableBoneParts.append(dict(name=chest.getName(), polygon=chestRect))

    waistRect = boneImg.templateMatch(waist)
    if waistRect != None:
      availableBoneParts.append(dict(name=waist.getName(), polygon=waistRect))

  boneImg.gray2bgr()

  boneImg.drawPolylines(craniumBySideRect)
  boneImg.drawPolylines(legRect)
  boneImg.drawPolylines(armRect)
  boneImg.drawPolylines(chestRect)
  boneImg.drawPolylines(waistRect)

  metastasisImg.gray2bgr()
  for i in range(0, len(features)):  # Iterate over detected metastasis
    # Draw centroid and approximate convex hull for each one of them
    metastasisImg.drawCircle(features[i]['centroid'])
    metastasisImg.drawContours([features[i]['convexHull']])
    isInside = False
    results.write('Detected metastasis of size ' + str(
        features[i]['area']) + ' located at ')
    for bonePart in availableBoneParts:
      if isPointsInsidePolygon(features[i]['convexHull'], bonePart['polygon']).all():
        results.write(' ' + bonePart['name'])
        isInside = True
    if isInside == False:
      results.write(' Unknown location')
    results.write('\n')

  boneImg.show()
  # metastasisImg.show()

results.close()
