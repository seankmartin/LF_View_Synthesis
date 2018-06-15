import inviwopy
from inviwopy.glm import vec3, normalize

from inviwo_maths import cross_product

class LightFieldCamera:
    def __init__(self, look_from, look_to, look_up = vec3(0, 1, 0), 
                 interspatial_distance = 1.0, spatial_rows = 8, spatial_cols = 8):
        """Create a light field camera array.

        Keyword arguments:
        look_from, look_to, look_up -- camera vectors for top left camera (default up is y-axis)
        interspatial_distance -- distance between cameras in array (default 1.0)
        spatial_rows, spatial_cols -- camera array dimensions (default 8 x 8)
        """
        self.set_look(look_from, look_to, look_up)
        self.spatial_rows = spatial_rows
        self.spatial_cols = spatial_cols

    def set_look(self, look_from, look_to, look_up):
        """Set the top left camera in the array to look_from, look_to and look_up"""
        self.look_from = normalize(look_from)
        self.look_to = normalize(look_to)
        self.look_up = normaliize(look_up)

    def preview_array(self):
        """Move the inviwo camera through the array for the current workspace"""
        # Save the current camera position
        prev_cam_look_from = cam.lookFrom
        prev_cam_look_to = cam.lookTo

        for (look_from, look_to) in self.calculate_camera_array():
            cam.lookFrom = look_from
            cam.lookTo = look_to
            inviwo_utils.update()

        # Reset the camera to original position
        cam.lookFrom = prev_cam_look_from
        cam.lookTo = prev_cam_look_to

    def get_look_right(self):
        """Get the right look vector for the top left camera"""
        view_direction = self.look_to - self.look_from
        return normalize(cross_product(self.look_up, self.view_direction))

    def calculate_camera_array(self):
        """Returns a list of (look_from, look_to) tuples for the full camera array"""
        look_list = []

        row_step_vec = self.look_up * interspatial_distance 
        col_step_vec = self.get_look_right() * interspatial_distance

        #Start at the top left camera position
        for i in range(spatial_rows):
            row_movement = row_step_vec * (-i)
            row_look_from = self.look_from + row_movement 
            row_look_to = self.look_to + row_movement

            for j in range(spatial_cols):
                col_movement = col_step_vec * j
                cam_look_from = row_look_from + col_movement
                cam_look_to = row_look_to + col_movement

                look_list.append((cam_look_from, cam_look_to))

        return look_list