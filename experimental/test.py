from pupil_apriltags import Detector, Detection
import math 
from scipy.optimize import minimize
import cv2
import numpy as np
import time
from copy import deepcopy


class Tag(Detection):
	def __init__(self, detection_obj):
		super().__init__()
		self = detection_obj
		self.age = 0

	def getDist(self):
		d1 = math.hypot(qrCode.corners[0][0]-qrCode.corners[1][0], qrCode.corners[0][1]-qrCode.corners[1][1])
		d2 = math.hypot(qrCode.corners[0][0]-qrCode.corners[-1][0], qrCode.corners[0][1]-qrCode.corners[-1][1])
		max_size = max(d1, d2)
		angle = (max_size / frame.shape[0]) * fov[1]
		return qrCodeRealSize / math.tan(angle) # opp / tan = adj

# https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
def polyArea(points):
	x = np.array([i[0] for i in points[0]])
	y = np.array([i[1] for i in points[0]])
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

# https://www.alanzucconi.com/2017/03/13/positioning-and-trilateration/
# Mean Square Error
# locations: [ (lat1, long1), ... ]
# distances: [ distance1, ... ]
def mse(x, locations, distances):
	mse = 0.0
	for location, distance in zip(locations, distances):
		distance_calculated = math.hypot(x[0]-location[0], x[1]-location[1])
		mse += math.pow(distance_calculated - distance, 2.0)
	return mse / len(distances)

def triangulatePos(locations, distances, first_guess=(0, 0)): # can pass initial guess from previous time-step
	# initial_location: (lat, long)
	# locations: [ (lat1, long1), ... ]
	# distances: [ distance1,     ... ] 
	result = minimize(
		mse,                         # The error function
		first_guess,            # The initial guess
		args=(locations, distances), # Additional parameters for mse
		method='L-BFGS-B',           # The optimisation algorithm
		options={
			'ftol':1e-5,         # Tolerance
			'maxiter': 1e+7      # Maximum iterations
		})
	# result.success
	return result.x

def getDist(qrCode, fov, qrCodeRealSize, frame): # passed frame so it can see the size
	# finds the distance of a single fiducial
	d1 = math.hypot(qrCode.corners[0][0]-qrCode.corners[1][0], qrCode.corners[0][1]-qrCode.corners[1][1])
	d2 = math.hypot(qrCode.corners[0][0]-qrCode.corners[-1][0], qrCode.corners[0][1]-qrCode.corners[-1][1])
	max_size = max(d1, d2)
	angle = (max_size / frame.shape[0]) * fov[1]
	return qrCodeRealSize / math.tan(angle) # opp / tan = adj

def createArrays(allLocations, dists):
	# turns the locations and fiducials distances dictionaries into two lists so they can be passed into the triangulator
	locations = []
	distances = []
	for loc in dists.keys():
		if loc in allLocations.keys():
			locations.append(allLocations[loc])
			distances.append(dists[loc])
	return distances, locations

def findDists(qrCodes, fov, realSize, frame):
	# creates a dict of all the seen fiducials and their distances from the camera
	dists = {}
	for qr in qrCodes.keys():
		dists[qr] = getDist(qrCodes[qr], fov, realSize, frame)
	return dists

detector = Detector(families='tag16h5',
					   nthreads=1,
					   quad_decimate=1.0,
					   quad_sigma=0.0,
					   refine_edges=1,
					   decode_sharpening=0.25,
					   debug=0)

def readQrCodes(image):
	tags = detector.detect(image, estimate_tag_pose=False, camera_params=None, tag_size=None)
	tags = map(Tag, tags)
	ret = {}
	for tag in tags:
		ret[tag.tag_id] = tag
	return ret

def removeFalsePositives(tags, n = 1):
	# remove any fiducials that have more than n errors in their hamming code
	# n = 0 almost never gets false positives but dosent work very well at distances
	# n = 1 increases consitency of seeing the fiducials at distance but has some false positives
	# n >= 2 largest amount of false positives
	newTags = {}
	for tag in tags.keys():
		if tags[tag].hamming <= n:
			newTags[tag] = tags[tag]
	return newTags

def main(image, real_locations, real_size, fov, prev_pos = (0, 0)):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	qrCodes = readQrCodes(gray)
	qrCodes = removeFalsePositives(qrCodes)

	dists_dict = findDists(qrCodes, fov, real_size, gray)
	cv2.imshow("feed", annotate(gray, dists_dict, qrCodes))

	if len(qrCodes) >= 1: # found a fiducial

		distances_arr, locations_arr = createArrays(real_locations, dists_dict) 
		# dosent include any fiducials that dont have corosponding positions in real_locations

		if len(distances_arr) >= 1 and len(locations_arr) >= 1: # found a fiducial we recognise
			pos = triangulatePos(locations_arr, distances_arr, (1, 1))
			return np.array(pos, dtype=float)
	return np.zeros(2, dtype=float)

def cornerDetection(img):
	# print("type", type(thresh_frame), "    shape",thresh_frame.shape)
	dst = cv2.cornerHarris(thresh_frame,2,3,0.04)

	#result is dilated for marking the corners, not important
	dst = cv2.dilate(dst,None)

	# Threshold for an optimal value, it may vary depending on the image.
	img[dst>0.01*dst.max()]=[0,0,255]

	return img

def annotate(img, dists, qrCodes):
	# draws circles and their id around the tags
	font = cv2.FONT_HERSHEY_SIMPLEX 
	for k in qrCodes.keys():
		size = 1/(dists[k]/20)
		pos = (int(qrCodes[k].center[0]), int(qrCodes[k].center[1]))
		cv2.circle(img, pos, int(size), 255, 5)
		cv2.putText(img, f"id:{qrCodes[k].tag_id}", pos, font, 1, 255, 2, cv2.LINE_AA) 
	return img

def drawMap(pos, locations, rang=(2.5, 2.5), size=(600, 800, 3)):
	# draws a map to visualise the triangulated position relative to the markers
	canvas = np.zeros(size)
	for k in locations.keys():
		cv2.circle(canvas, ( int(locations[k][0]/rang[0] * size[1]), int(locations[k][1]/rang[1] * size[0])), 5, (255, 0, 0), 5)
	cv2.circle(canvas, (int(pos[0]/rang[0] * size[1]), int(pos[1]/rang[1] * size[0])), 15, (255, 255, 255), 10)
	return canvas

if __name__ == "__main__":
	fov = (math.radians(68), math.radians(40)) # my laptop camera

	img = cv2.imread("tag16h5_multiple.png")

	cap = cv2.VideoCapture(0)
	real_locations = {0:(0, 0.38), 1:(0, 0.44), 2:(0, 0.38), 3:(0, 0.44),
	4:(0.61, 0), 6:(0.61, 0), 5:(0.665, 0), 7:(0.665, 0),
	26:(1.12, 0), 27:(1.12, 0), 28:(1.18, 0), 29:(1.18, 0)}
	real_size = 0.048

	pos = np.zeros(2, dtype=float)
	pos_smoothing = 0.2 # among of pos to be the new pos
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()

		# newImg = cornerDetection(img)
		pos = main(frame, real_locations, real_size, fov, pos)*pos_smoothing + pos*(1 - pos_smoothing)
		cv2.imshow("map", drawMap(pos, real_locations))

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()