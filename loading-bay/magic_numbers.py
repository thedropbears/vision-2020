import numpy as np

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

INNER_OUTER_RATIO = 11 / 3
INNER_OUTER_ERROR = 0.5

RAW_RECT_AREA_ERROR = 0.3

MIN_CONTOUR_AREA = 40

# Recognition parameters. These should be variables that come from calibration.
HSV_LOWER_BOUND = (60, 50, 50)
HSV_UPPER_BOUND = (90, 255, 255)

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
