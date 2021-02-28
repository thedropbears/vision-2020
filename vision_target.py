import numpy as np

class VisionTarget:
    def __init__(self, contour: np.ndarray) -> None:
        """Initialise a vision target object

        Args:
            contour: a single numpy/opencv contour
        """
        self.contour = contour.reshape(-1, 2)
        self._validate_and_reduce_contour()

    def _validate_and_reduce_contour(self):
        self.is_valid_target = True

    def get_leftmost_x(self) -> int:
        return min(list(self.contour[:, 0]))

    def get_rightmost_x(self) -> int:
        return max(list(self.contour[:, 0]))

    def get_middle_x(self) -> int:
        return (self.get_rightmost_x() + self.get_leftmost_x()) / 2

    def get_middle_y(self) -> int:
        return (self.get_lowest_y() + self.get_highest_y()) / 2

    def get_highest_y(self) -> int:
        return min(list(self.contour[:, 1]))

    def get_lowest_y(self) -> int:
        return max(list(self.contour[:, 1]))

    def get_height(self) -> int:
        return self.get_lowest_y() - self.get_highest_y()

    def get_width(self) -> int:
        return self.get_rightmost_x() - self.get_leftmost_x()