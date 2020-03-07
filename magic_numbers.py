import numpy as np
import math

# Magic Numbers:

# The following were careful measurements of the frame area with the camera
# aimed at a flat wall, and the distance of the camera from the wall. All are in
# millimetres.
FOV_WIDTH = 1793
FOV_HEIGHT = 2303
FOV_DISTANCE = 2234

MAX_FOV_WIDTH = math.atan2(FOV_WIDTH / 2, FOV_DISTANCE) * 2  # 54.54 degrees
MAX_FOV_HEIGHT = math.atan2(FOV_HEIGHT / 2, FOV_DISTANCE) * 2  # 42.31 degrees

MAX_ZOOM = 200

# Camera settings
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

CAMERA_HEIGHT = 0.913  # in metres, off the ground
TARGET_HEIGHT_BOTTOM = 2.064  # in metres, off the ground
TARGET_HEIGHT_TOP = 2.500  # in metres, off the ground (nominally 2.496 per specs)
TILT_CORRECTION = 2.289  # Degrees, measured delta at 10 metres. Provisional.
GROUND_ANGLE = math.radians(16.5 + TILT_CORRECTION)  # Camera tilt, actually

# Recognition parameters. These should be variables that come from calibration.
HSV_LOWER_BOUND = (60, 50, 30)
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

MIN_CONTOUR_AREA = 50
LOADING_INNER_OUTER_RATIO = 11 / 3
LOADING_RECT_AREA_RATIO = 0.2

# Camera parameters (fixed per-camera at a given zoom) These are for the
# Logitechs
FOCAL_LENGTH = 3.67  # mm
SENSOR_WIDTH = 4.8  # mm
SENSOR_HEIGHT = 3.6  # mm
PP_MIN_CONTOUR_AREA = 50
PP_MAX_CONTOUR_AREA = 1300
PP_MIN_AREA_RATIO = 0.175
PP_MAX_AREA_RATIO = 0.4

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

# How much of the returned angle should be the last returned one
ANGLE_SMOOTHING_AMOUNT = 0.3

DIST_SMOOTHING_AMOUNT = 0.8

# Recognition parameters. These should be variables that come from calibration.
LOADING_BAY_HSV_LOWER_BOUND = (60, 50, 50)
LOADING_BAY_HSV_UPPER_BOUND = (90, 255, 255)

# (0)_______(3)
#  | (4)_(7) |
#  |  |   |  |
#  |  |   |  |
#  | (5)_(6) |
# (1)_______(2)

LOADING_BAY_POINTS = (
    np.array(
        [
            [3.5, 5.5, 0.0],
            [3.5, -5.5, 0.0],
            [-3.5, -5.5, 0.0],
            [-3.5, 5.5, 0.0],
            [1.5, 3.5, 0.0],
            [1.5, -3.5, 0.0],
            [-1.5, -3.5, 0.0],
            [-1.5, 3.5, 0.0],
        ],
        dtype=np.float32,
    )
    * 0.0254
)

INNER_OUTER_ERROR = 0.5

RAW_RECT_AREA_ERROR = 0.3
