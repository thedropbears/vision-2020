from dt_apriltags import Detector, Detection
import math 
from scipy.optimize import minimize
import cv2
import numpy as np
import time
from copy import deepcopy
import magic_numbers

class FiducialVision:
	ap_tag = {}
	max_age = 50
	min_age = 5
	age_decriment = 5
	age_incriment = 1

	fov = (magic_numbers.MAX_FOV_WIDTH, magic_numbers.MAX_FOV_HEIGHT)
	frame_size = (640, 480)

	def __init__(self, real_locations, real_size):
		self.real_locations = real_locations
		self.real_size = real_size

		self.detector = Detector(families='tag16h5')
		self.dists_dict = {}
		self.ages = {}

		self.pos = [0, 0]


	def run(self, img):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		self.ap_tags = self._findApTags(gray)
		self.ap_tags = self._removeFalsePositives(self.ap_tags)

		self.dists_dict.update(self._findDists()) # tracks the last known distance for each tag
		# if we stop seeing a april tag its last known location is kept in this dict, chcek self.ages to see if it should be used


		if len(self.ap_tags) >= 1: # found one or more april tags
			self.ages = self._trackAges()

			distances_arr, locations_arr = self._createArrays()
			if len(distances_arr) >= 1:
				self.pos = self._triangulate(distances_arr, locations_arr)
				self.pos = [ max(self.pos[0], 0), max(self.pos[1], 0) ]
				return self.pos
		return None

	def _trackAges(self):
		for key in self.ap_tags.keys():
			if key in self.ages.keys(): # saw an existing april tag this frame
				if self.ages[key] < self.max_age: # and it hasent reached the max age yet
					self.ages[key] = max(self.ages[key], 0) + self.age_incriment
			else: # saw a new april tag this frame
				self.ages[key] = 0

		for key in self.ages.keys():
			if key not in self.ap_tags.keys(): # didnt see an existing april tag this frame
				self.ages[key] -= self.age_decriment
		return self.ages


	def _findApTags(self, img):
		tags = self.detector.detect(img, estimate_tag_pose=False, camera_params=None, tag_size=None)
		# tags = map(Tag, tags)
		ret = {}
		for tag in tags:
			ret[tag.tag_id] = tag
		return ret

	def _removeFalsePositives(self, tags, n = 1):
		# remove any fiducials that have more than n errors in their hamming code
		# n = 0 almost never gets false positives but dosent work very well at distances
		# n = 1 increases consitency of seeing the fiducials at distance but has some false positives
		# n >= 2 largest amount of false positives
		newTags = {}
		for tag in tags.keys():
			if tags[tag].hamming <= n:
				newTags[tag] = tags[tag]
		return newTags

	def _findDists(self):
		# creates a dict of all the seen fiducials and their distances from the camera
		dists = {}
		for qr in self.ap_tags.keys():
			mid_offset = math.hypot( self.frame_size[0]/2-self.ap_tags[qr].center[0],
				self.frame_size[1]/2-self.ap_tags[qr].center[1])
			dists[qr] = self._getDist(self.ap_tags[qr])
		return dists

	def _getDist(self, qrCode): # passed frame so it can see the size
		# finds the distance of a single fiducial
		d1 = math.hypot(qrCode.corners[0][0]-qrCode.corners[1][0], qrCode.corners[0][1]-qrCode.corners[1][1])
		d2 = math.hypot(qrCode.corners[0][0]-qrCode.corners[-1][0], qrCode.corners[0][1]-qrCode.corners[-1][1])
		max_size = max(d1, d2)
		angle = (max_size / self.frame_size[1]) * self.fov[1]
		return self.real_size / math.tan(angle) # opp / tan = adj

	def _createArrays(self):
		# turns the locations and fiducials distances dictionaries into two lists so they can be passed into the triangulator
		locations = []
		distances = []
		used_tags = []
		for key in self.dists_dict.keys():
			if key in self.real_locations.keys() and self.ages[key] > self.min_age:
				locations.append(self.real_locations[key])
				distances.append(self.dists_dict[key])
				used_tags.append(key)
		print(f"used tags {used_tags}")
		return distances, locations

	@staticmethod
	def _mse(x, locations, distances):
		mse = 0.0
		for location, distance in zip(locations, distances):
			distance_calculated = math.hypot(x[0]-location[0], x[1]-location[1])
			mse += math.pow(distance_calculated - distance, 2.0)
		return mse / len(distances)

	def _triangulate(self, distances, locations): # can pass initial guess from previous time-step
		# initial_location: (lat, long)
		# locations: [ (lat1, long1), ... ]
		# distances: [ distance1,     ... ] 
		result = minimize(
			self._mse,                       # The error function
			self.pos,            # The initial guess
			args=(locations, distances), # Additional parameters for mse
			method='L-BFGS-B',           # The optimisation algorithm
			options={
				'ftol':1e-5,         # Tolerance
				'maxiter': 1e+7      # Maximum iterations
			})
		# result.success
		return result.x


	def annotate(self, img):
		# draws circles and their id around the tags
		font = cv2.FONT_HERSHEY_SIMPLEX 
		for k in self.ap_tags.keys():
			size = 1/(self.dists_dict[k]/20)
			pos = (int(self.ap_tags[k].center[0]), int(self.ap_tags[k].center[1]))
			cv2.circle(img, pos, int(abs(size)), (255, 255, 255), 5)
			fid_id = self.ap_tags[k].tag_id
			cv2.putText(img, f"d:{self.dists_dict[k]}, id:{fid_id}", pos, font, 0.5, (255, 255, 255), 2, cv2.LINE_AA) 
		return img

	def drawMap(self, locations, rang=(2.5, 2.5), size=(600, 800, 3)):
		# draws a map to visualise the triangulated position relative to the markers
		canvas = np.zeros(size)
		for k in locations.keys():
			cv2.circle(canvas, ( int(locations[k][0]/rang[0] * size[1]), int(locations[k][1]/rang[1] * size[0])), 5, (255, 0, 0), 5)
		cv2.circle(canvas, (int(self.pos[0]/rang[0] * size[1]), int(self.pos[1]/rang[1] * size[0])), 15, (255, 255, 255), 10)
		return canvas

if __name__ == "__main__":
	locations = {0:(0, 0.38), 1:(0, 0.44), 2:(0, 0.38), 3:(0, 0.44),
	4:(0.61, 0), 6:(0.61, 0), 5:(0.665, 0), 7:(0.665, 0),
	26:(1.12, 0), 27:(1.12, 0), 28:(1.18, 0), 29:(1.18, 0)}
	real_size = 0.05 # meters
	fidVision = FiducialVision(real_locations = locations, real_size = real_size)

	cap = cv2.VideoCapture(2)
	while True:
		ret, frame = cap.read()
		fidVision.run(frame)
		cv2.imshow("map", fidVision.drawMap(locations))
		cv2.imshow("feed", fidVision.annotate(frame))

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break