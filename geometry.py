from math import pow, sqrt
import matplotlib.path as mpltPath

def euclidianDistance(p1, p2):
  return sqrt((pow((p2[0] - p1[0]), 2)) + (pow((p2[1] - p1[1]), 2)))

def isPointInsidePolygon(point, polygon):
  path = mpltPath.Path(polygon)
  return path.contains_point(point)

def isPointsInsidePolygon(points, polygon):
  path = mpltPath.Path(polygon)
  return path.contains_points(points)

def distanceToPolygon(p, polygon):
  minDistance = euclidianDistance(p, polygon[0])
  for i in range(1, len(polygon)):
    distance = euclidianDistance(p, polygon[i])
    if distance < minDistance:
      minDistance = distance
  return minDistance
