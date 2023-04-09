import numpy as np


class Road:

    def __init__(
            self,
            center: np.array = None
    ) -> None:
        if center is None:
            theta = np.linspace(0, 2 * np.pi, 100)
            radius = 5
            center = [0, 0]

            x = radius * np.cos(theta) + center[0]
            y = radius * np.sin(theta) + center[1]
            y += 5
            self.centerline = np.stack((x, y), axis=1)
        else:
            self.centerline = center

    def wrap_to_pi(
            self,
            angle
    ):
        """Wrap an angle in radians to the range [-pi, pi].

        Args:
            angle (double): angle in radians.

        Returns:
            double: angle in the range of -pi to pi.
        """
        return np.mod(angle + np.pi, 2 * np.pi) - np.pi

    def find_nearest_point(
            self,
            vehicle_position: np.ndarray = None
    ):
        """Find the nearest point on centerline to vehicle position.

        Args:
            vehicle_position (np.ndarray): array of shape (2,) representing the vehicle position.

        Returns:
            tuple: index of the nearest point on centerline and nearest point coordinates.
        """
        dist = np.linalg.norm(self.centerline - vehicle_position, axis=1)
        idx = np.argmin(dist)
        return idx, self.centerline[idx]

    def compute_errors(self, vehicle_position, vehicle_heading):
        """Compute cross-track error and heading error.

        Args:
            vehicle_position (np.ndarray): array of shape (2,) representing the vehicle position.
            vehicle_heading (float): the vehicle heading in radians.

        Returns:
            tuple: cross-track error (float) and heading error (float) in radians.
        """
        # find the nearest point on centerline
        idx, nearest_point = self.find_nearest_point(vehicle_position)

        # calculate cross-track error
        v_vec = vehicle_position - self.centerline[idx - 1]
        w_vec = nearest_point - self.centerline[idx - 1]
        cte = np.cross(v_vec, w_vec) / np.linalg.norm(w_vec)

        # calculate heading error
        desired_heading = np.arctan2(self.centerline[idx + 1][1] - self.centerline[idx][1],
                                     self.centerline[idx + 1][0] - self.centerline[idx][0])
        heading_error = self.wrap_to_pi(desired_heading - vehicle_heading)

        # calculate positional error
        next_point = self.centerline[min(idx + 1, len(self.centerline) - 1)]
        v_vec_next = vehicle_position - nearest_point
        w_vec_next = next_point - nearest_point
        pos_error = np.cross(v_vec_next, w_vec_next) / np.linalg.norm(w_vec_next)

        return cte, heading_error, pos_error
