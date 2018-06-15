import os

import inviwopy

from Camera import LightFieldCamera

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

    #Get the names of the canvases in the workspace
    for canvas in network.canvases:
        print(canvas.displayName)
    
    lf_camera_here = LightFieldCamera(cam.lookFrom, cam.lookTo)
    lf_camera_here.preview_array()
    

if __name__ == '__main__':
    main()