# Our imports
from Dicom import Dicom
from os import listdir
from os.path import isfile, join
from Image import Image
from geometry import isPointsInsidePolygon

# Initialize instances data
craniumBySide = Image(Image.read(
    './bonePartsInstances/cranium.png', 'gray'), 'cranium')
legs = Image(Image.read(
    './bonePartsInstances/legs.png', 'gray'), 'legs')
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
      [7, 100, (255, 0, 0)], [50, 300, (0, 0, 255)]])

  boneImg = Image(segmentedBGR, "Bone")
  metastasisImg = Image(segmentedBGR, "Metastasis")

  defaultImage = Image(segmentedBGR, 'default')

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
  legsRect = boneImg.templateMatch(legs)

  metastasisImg.findCountours(1)
  features = metastasisImg.findContoursFeatures()

  if len(features) == 0:
    results.write('We did not detected any metastasis\n')

  availableBoneParts = []

  # Here we split the bone image into bone parts, so we can discover witch one of them is suffering from metastasis
  # In this code, we are using instance detection using SIFT and homography or template matching by Normalized cross correlation

  craniumBySideRect = boneImg.templateMatch(craniumBySide)
  if craniumBySideRect != None:
    availableBoneParts.append(
        dict(name=craniumBySide.getName(), polygon=craniumBySideRect))

  chestRect = boneImg.templateMatch(chest)
  if chestRect != None:
    bottom_left, top_left, top_right, bottom_right = chestRect
    halfSize = (top_right[0] - top_left[0])/2
    rightChestRect = [bottom_left, top_left, [top_left[0] + halfSize,
                                              top_right[1]], [bottom_left[0] + halfSize, bottom_right[1]]]
    leftChestRect = [[bottom_left[0] + halfSize, bottom_left[1]],
                     [top_left[0] + halfSize, top_left[1]], top_right, bottom_right]
    availableBoneParts.append(
        dict(name='rightChest', polygon=rightChestRect))
    availableBoneParts.append(
        dict(name='leftChest', polygon=leftChestRect))

  rightArmRect = boneImg.templateMatch(arm)
  if rightArmRect != None and chestRect != None:
    rightArm_bottom_left, rightArm_top_left, rightArm_top_right, rightArm_bottom_right = rightArmRect
    chest_bottom_left, chest_top_left, chest_top_right, chest_bottom_right = chestRect
    w = rightArm_top_right[0] - rightArm_top_left[0]
    leftArmRect = [[chest_bottom_right[0], rightArm_bottom_left[1]], [chest_top_right[0], rightArm_top_left[1]], [chest_top_right[0] + w, rightArm_top_right[1]], [
        chest_bottom_right[0] + w, rightArm_bottom_right[1]]]
    availableBoneParts.append(
        dict(name='rightArm', polygon=rightArmRect))
    availableBoneParts.append(
        dict(name='leftArm', polygon=leftArmRect))

  waistRect = boneImg.templateMatch(waist)
  if waistRect != None:
    availableBoneParts.append(dict(name=waist.getName(), polygon=waistRect))

  legsRect = boneImg.templateMatch(legs)
  if legsRect != None:
    bottom_left, top_left, top_right, bottom_right = legsRect
    halfSize = (top_right[0] - top_left[0])/2
    rightLegRect = [bottom_left, top_left, [top_left[0] + halfSize,
                                            top_right[1]], [bottom_left[0] + halfSize, bottom_right[1]]]
    leftLegRect = [[bottom_left[0] + halfSize, bottom_left[1]],
                   [top_left[0] + halfSize, top_left[1]], top_right, bottom_right]
    availableBoneParts.append(
        dict(name='rightLeg', polygon=rightLegRect))
    availableBoneParts.append(
        dict(name='leftLeg', polygon=leftLegRect))

  # Draw boneParts rects and names to debug
  for bonePart in availableBoneParts:
    defaultImage.drawText(
        bonePart['name'], int(bonePart['polygon'][0][0]), int(bonePart['polygon'][0][1] - 20), (0, 255, 0), 2, 0.6)
    defaultImage.drawPolylines(bonePart['polygon'])

  metastasisImg.gray2bgr()
  for i in range(0, len(features)):  # Iterate over detected metastasis
    # Draw centroid and approximate convex hull for each one of them; Export features to results.txt file
    metastasisImg.drawCircle(features[i]['centroid'])
    metastasisImg.drawContours([features[i]['convexHull']])
    isInside = False
    results.write('Detected metastasis of size ' + str(
        features[i]['area']) + ' located at ')
    for bonePart in availableBoneParts:
      # Check fot bonePart location
      if isPointsInsidePolygon(features[i]['convexHull'], bonePart['polygon']).all():
        results.write(bonePart['name'] + ' ')
        isInside = True
    if isInside == False:
      results.write('unknown location')
    results.write('(' + str(features[i]['centroid'][0]) +
                  ',' + str(features[i]['centroid'][1]) + ')' + '\n')

  defaultImage.show()
  # boneImg.show()
  # metastasisImg.show()

results.close()
