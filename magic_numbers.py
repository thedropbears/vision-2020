import numpy as np
import math

# Magic Numbers:

# Camera parameters (fixed per-camera at a given zoom) These are for the
# Logitechs
FOCAL_LENGTH = 3.67  # mm
SENSOR_WIDTH = 4.8  # mm
SENSOR_HEIGHT = 3.6  # mm

MAX_FOV_WIDTH = 1.158375  # 66.37
MAX_FOV_HEIGHT = 0.91193453  # 52.25

MAX_ZOOM = 200

# Camera settings
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Recognition parameters. These should be variables that come from calibration.
HSV_LOWER_BOUND = (60, 50, 15)
HSV_UPPER_BOUND = (90, 255, 255)

# Target shape parameters
# Order of Power Port points
# (0)__    (0, 0)    __(3)
#    \ \            / /
#     \ \          / /
#      \ \________/ /
#      (1)________(2)

PORT_DIMENTIONS = [0.993, 43.18]

PORT_POINTS = [  # Given in inches as per the manual
    [19.625, 0, 0],
    [19.625 / 2, -17, 0],
    [-19.625 / 2, -17, 0],
    [-19.625, 0, 0],
]
PORT_POINTS = np.array(  # Converted to mm
    [(2.54 * i[0], 2.54 * i[1], 0) for i in PORT_POINTS], np.float32
).reshape((4, 1, 3))

MIN_CONTOUR_AREA = 40
LOADING_INNER_OUTER_RATIO = 3.62
LOADING_RECT_AREA_RATIO = 0.2

POWER_PORT_AREA_RATIO = 0.2


FX = FOCAL_LENGTH * FRAME_WIDTH / SENSOR_WIDTH
FY = FOCAL_LENGTH * FRAME_HEIGHT / SENSOR_HEIGHT
CX = FRAME_WIDTH / 2
CY = FRAME_HEIGHT / 2
INTR_MATRIX = np.array(
    [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=np.float32
)
DIST_COEFF = np.array([0, 0, 0, 0], dtype=np.float32)

# The intrinsic matrix for the second Logitech C920.
# Found by calibration.
C920_2_INTR_MATRIX = np.array(
    [
        [310.6992514, 0.0, 152.13193831],
        [0.0, 312.88049348, 120.88875952],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

# The distortion coefficients for the second Logitech C920.
# Found by calibration.
C920_2_DIST_COEFFS = np.array(
    [[0.13840045, -0.3277049, -0.00142985, -0.00095689, 0.28607425]], dtype=np.float32
)

CAMERA_HEIGHT = 0.66
TARGET_HEIGHT = 2.04
GROUND_ANGLE = math.radians(30)
