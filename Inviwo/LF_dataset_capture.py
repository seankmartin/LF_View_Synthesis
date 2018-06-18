import os
from time import sleep

import inviwopy
import ivw.utils as inviwo_utils
from inviwopy.glm import vec3, normalize

def cross_product(vec_1, vec_2):
    result =  vec3(
        (vec_1.y * vec_2.z) - (vec_1.z * vec_2.y),
        (vec_1.z * vec_2.x) - (vec_1.x * vec_2.z),
        (vec_1.x * vec_2.y) - (vec_1.y * vec_2.x))
    return result

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
        self.interspatial_distance = interspatial_distance

    def set_look(self, look_from, look_to, look_up):
        """Set the top left camera in the array to look_from, look_to and look_up"""
        self.look_from = look_from
        self.look_to = look_to
        self.look_up = look_up
    
    def get_row_col_number(self, index):
        row_num = index // self.spatial_cols
        col_num = index % self.spatial_cols
        return row_num, col_num
    
    def preview_array(self, cam):
        """Move the inviwo camera through the array for the current workspace"""
        # Save the current camera position
        prev_cam_look_from = cam.lookFrom
        prev_cam_look_to = cam.lookTo

        for idx, (look_from, look_to) in enumerate(self.calculate_camera_array()):
            cam.lookFrom = look_from
            cam.lookTo = look_to
            row_num, col_num = self.get_row_col_number(idx)
            print("Viewing position ({}, {})".format(row_num, col_num))
            inviwo_utils.update()
            sleep(0.1)

        # Reset the camera to original position
        cam.lookFrom = prev_cam_look_from
        cam.lookTo = prev_cam_look_to

    def get_look_right(self):
        """Get the right look vector for the top left camera"""
        view_direction = self.look_to - self.look_from
        right_vec = normalize(cross_product(view_direction, self.look_up))
        return right_vec

    def calculate_camera_array(self):
        """Returns a list of (look_from, look_to) tuples for the full camera array"""
        look_list = []

        row_step_vec = normalize(self.look_up) * self.interspatial_distance 
        col_step_vec = self.get_look_right() * self.interspatial_distance

        #Start at the top left camera position
        for i in range(self.spatial_rows):
            row_movement = row_step_vec * (-i)
            row_look_from = self.look_from + row_movement 
            row_look_to = self.look_to + row_movement

            for j in range(self.spatial_cols):
                col_movement = col_step_vec * j
                cam_look_from = row_look_from + col_movement
                cam_look_to = row_look_to + col_movement
                
                look_list.append((cam_look_from, cam_look_to))
        
        return look_list

#Courtesy of Sean Bruton
def get_sub_dir_for_saving(base_dir):
    num_sub_dirs = sum(os.path.isdir(os.path.join(dataset_dir, el))
                   for el in os.listdir(dataset_dir))

    sub_dir_to_save_to_name = str(num_sub_dirs)
    sub_dir_to_save_to_name = sub_dir_to_save_to_name.zfill(4)

    sub_dir_to_save_to = os.path.join(dataset_dir, sub_dir_to_save_to_name)
    os.mkdir(sub_dir_to_save_to)

    return sub_dir_to_save_to

def main():
    #Setup
    app = inviwopy.app
    network = app.network
    cam = network.EntryExitPoints.camera
    #cam.lookTo = vec3(0, 0, 0)
    #cam.lookUp = vec3(0, 1, 0)
    
    #Get the names of the canvases in the workspace
    for canvas in network.canvases:
        print(canvas.displayName)
    
    print("camera properties")
    print(cam.lookTo, cam.LookFrom)
    lf_camera_here = LightFieldCamera(cam.lookFrom, cam.lookTo, interspatial_distance = 0.5)
    print("lf properties")
    print(lf_camera_here.look_from, lf_camera_here.look_to)
    lf_camera_here.preview_array(cam)
    

if __name__ == '__main__':
    main()