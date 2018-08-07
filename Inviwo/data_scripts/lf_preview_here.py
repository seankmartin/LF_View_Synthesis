#Inviwo Python script 
import inviwopy
from inviwopy.glm import vec3, ivec2
import ivw.utils as inviwo_utils

from lf_camera import LightFieldCamera

def main(pixel_dim):
    #Setup
    app = inviwopy.app
    network = app.network
    cam = network.EntryExitPoints.camera
    cam.lookUp = vec3(0, 1, 0)
    cam.nearPlane = 6.0
    cam.farPlane = 1000.0
    canvases = inviwopy.app.network.canvases
    for canvas in canvases:
        canvas.inputSize.dimensions.value = ivec2(pixel_dim, pixel_dim)
    inviwo_utils.update()

    # Create a light field camera at the current camera position
    lf_camera_here = LightFieldCamera(cam.lookFrom, cam.lookTo,
                                                               interspatial_distance=0.5)

    #Preview the lf camera array
    lf_camera_here.view_array(cam, save=False, should_time=True)


if __name__ == '__main__':
    PIXEL_DIM = 512
    
    main(PIXEL_DIM)
