"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d


class Algorithms:
    def __init__(self):
        self.scored_directional = {}
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range+1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        """INSERT YOUR CODE HERE"""
        x_pad = (win_size//2) + 1 + dsp_range
        y_pad = (win_size//2) + 1
        kernel = np.ones((win_size, win_size))
        left_image = np.pad(left_image, ((y_pad,), (x_pad,), (0,)), 'constant', constant_values=((0,0), (0,0), (0,0)))
        for D in range(len(disparity_values)):
            s = 0
            d = D - dsp_range
            temp = np.pad(right_image, ((y_pad, y_pad), (x_pad - d, x_pad + d), (0, 0)), 'constant', constant_values=((0,0), (0,0), (0,0)))
            ssd_image = (left_image - temp)**2
            for channel in range(ssd_image.shape[2]):
                s += convolve2d(ssd_image[y_pad:-y_pad, x_pad:-x_pad, channel], kernel, mode='same')
            ssdd_tensor[:, :, D] = s

        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        # you can erase the label_no_smooth initialization.
        # label_no_smooth = np.zeros((ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
        """INSERT YOUR CODE HERE"""
        label_no_smooth = np.argmin(ssdd_tensor, axis=2)
        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        if c_slice.ndim == 1 or c_slice.shape[1] == 1:
            return c_slice
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))
        """INSERT YOUR CODE HERE"""
        l_slice[:, 0] = c_slice[:, 0]
        for col in range(1, num_of_cols):
            min_score = min(l_slice[:, col - 1])
            for d in range(num_labels):
                a = l_slice[d, col - 1]
                if d == 0:
                    b = p1 + l_slice[d + 1, col - 1]
                elif d == num_labels-1:
                    b = p1 + l_slice[d - 1, col - 1]
                else:
                    b = p1 + min(l_slice[d - 1, col - 1], l_slice[d + 1, col - 1])
                if d < 2:
                    c = p2 + min(l_slice[d+2:, col-1])
                elif d >= num_labels - 2:
                    c = p2 + min(l_slice[:d-1, col-1])
                else:
                    c = p2 + min(min(l_slice[d+2:, col-1]), min(l_slice[:d-1, col-1]))
                M = min(a, b, c)
                l_slice[d, col] = c_slice[d, col] + M - min_score
        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        for row in range(ssdd_tensor.shape[0]):
            l[row] = Algorithms.dp_grade_slice(ssdd_tensor[row].T, p1, p2).T
        return Algorithms.naive_labeling(l)

    @staticmethod
    def extract_slice_by_direction(ssdd_tensor: np.ndarray, slice_index: int, direction: int) -> np.ndarray:
        if direction % 4 == 0:
            extracted_slice = np.fliplr(ssdd_tensor).diagonal(slice_index)
        elif direction % 4 == 1:
            extracted_slice = ssdd_tensor[slice_index].T
        elif direction % 4 == 2:
            extracted_slice = ssdd_tensor.diagonal(slice_index)
        elif direction % 4 == 3:
            extracted_slice = ssdd_tensor[:, slice_index].T
        if direction > 4:
            return np.fliplr(extracted_slice)
        return extracted_slice

    @staticmethod
    def get_score_route(h: int, w: int, direction: int):
        if direction % 4 == 0 or direction % 4 == 2:
            return range(-h+1, w)
        elif direction % 4 == 1:
            return range(h)
        return range(w)

    @staticmethod
    def calculate_score_per_direction(ssdd_tensor: np.ndarray, direction: int, p1: float, p2: float) -> np.ndarray:
        h, w = ssdd_tensor.shape[0], ssdd_tensor.shape[1]
        directed_score = np.zeros_like(ssdd_tensor)
        for r in Algorithms.get_score_route(h, w, direction):
            extracted_slice = Algorithms.extract_slice_by_direction(ssdd_tensor, r, direction)
            position = np.arange(extracted_slice.shape[1])
            score = Algorithms.dp_grade_slice(extracted_slice, p1, p2)
            if direction > 4:
                score = np.fliplr(score)
            if direction % 4 == 0:
                if r < 0:
                    i, j = position + abs(r), w - 1 - position
                else:
                    i, j = position, w - 1 - abs(r) - position
            elif direction % 4 == 1:
                i, j = r, position #np.arange(w)
            elif direction % 4 == 2:
                if r < 0:
                    i, j = position+abs(r), position
                else:
                    i, j = position, position+abs(r)
            elif direction % 4 == 3:
                i, j = position, r #np.arange(h)
            directed_score[i, j] = score.T
        return directed_score

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        """INSERT YOUR CODE HERE"""
        for i in range(1, num_of_directions + 1):
            self.scored_directional[i] = Algorithms.calculate_score_per_direction(ssdd_tensor, i, p1, p2)
            direction_to_slice[i] = Algorithms.naive_labeling(self.scored_directional[i])
        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        if self.scored_directional != {}:
            for i in range(1, num_of_directions+1):
                l += self.scored_directional[i]
            self.scored_directional.clear()
        else:
            for i in range(1, num_of_directions+1):
                l += Algorithms.calculate_score_per_direction(ssdd_tensor, i, p1, p2)
        return Algorithms.naive_labeling(l/num_of_directions)


class Bonus:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int, p: int = 1) -> np.ndarray: #_norm_p
        try:
            assert p >= 1, "Norm function must follow the condition p >= 1"
        except AssertionError as e:
            print(e)
            exit()
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range + 1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        x_pad = (win_size // 2) + 1 + dsp_range
        y_pad = (win_size // 2) + 1
        left_image = np.pad(left_image, ((y_pad,), (x_pad,), (0,)), 'constant',
                            constant_values=((0, 0), (0, 0), (0, 0)))
        for D in range(len(disparity_values)):
            s = 0
            d = D - dsp_range
            temp = np.pad(right_image, ((y_pad, y_pad), (x_pad - d, x_pad + d), (0, 0)), 'constant',
                          constant_values=((0, 0), (0, 0), (0, 0)))
            ssd_image = abs((left_image - temp)) ** p
            kernel = np.ones((win_size, win_size))
            for channel in range(ssd_image.shape[2]):
                s += convolve2d(ssd_image[y_pad:-y_pad, x_pad:-x_pad, channel], kernel, mode='same')
            ssdd_tensor[:, :, D] = s**(1/p)

        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, k: float, p: float = 0.5) -> np.ndarray:
        if c_slice.ndim == 1 or c_slice.shape[1] == 1:
            return c_slice
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))

        l_slice[:, 0] = c_slice[:, 0]
        for col in range(1, num_of_cols):
            min_score = min(l_slice[:, col - 1])
            for d in range(num_labels):
                line = l_slice[:, col-1]
                D = np.arange(num_labels)
                D = k*abs(D-d)**p
                line += D
                M = min(line)
                l_slice[d, col] = c_slice[d, col] + M - min_score
        return l_slice

    @staticmethod
    def calculate_score_per_direction(ssdd_tensor: np.ndarray, direction: int, k: float, p: float) -> np.ndarray:
        h, w = ssdd_tensor.shape[0], ssdd_tensor.shape[1]
        directed_score = np.zeros_like(ssdd_tensor)
        for r in Algorithms.get_score_route(h, w, direction):
            extracted_slice = Algorithms.extract_slice_by_direction(ssdd_tensor, r, direction)
            position = np.arange(extracted_slice.shape[1])
            score = Bonus.dp_grade_slice(extracted_slice, k, p)
            if direction > 4:
                score = np.fliplr(score)
            if direction % 4 == 0:
                if r < 0:
                    i, j = position + abs(r), w - 1 - position
                else:
                    i, j = position, w - 1 - abs(r) - position
            elif direction % 4 == 1:
                i, j = r, position #np.arange(w)
            elif direction % 4 == 2:
                if r < 0:
                    i, j = position+abs(r), position
                else:
                    i, j = position, position+abs(r)
            elif direction % 4 == 3:
                i, j = position, r #np.arange(h)
            directed_score[i, j] = score.T
        return directed_score

    def dp_labeling(self, ssdd_tensor: np.ndarray, k: float, p: float) -> np.ndarray:
        l = np.zeros_like(ssdd_tensor)
        for row in range(ssdd_tensor.shape[0]):
            l[row] = Bonus.dp_grade_slice(ssdd_tensor[row].T, k, p).T
        return Algorithms.naive_labeling(l)

    def dp_labeling_per_direction(self, ssdd_tensor: np.ndarray, k: float, p: float) -> dict:
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        for i in range(1, num_of_directions + 1):
            direction_to_slice[i] = Algorithms.naive_labeling(Bonus.calculate_score_per_direction(ssdd_tensor, i, k, p))
        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, k: float, p: float):
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        for i in range(1, num_of_directions+1):
            l += Bonus.calculate_score_per_direction(ssdd_tensor, i, k, p)
            print(f'L{i} finished')
        return Algorithms.naive_labeling(l/num_of_directions)

