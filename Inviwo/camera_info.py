#Inviwo Python script
import inviwopy
from inviwopy.glm import vec3

app = inviwopy.app
network = app.network

cam = network.EntryExitPoints.camera
#print(dir(cam))

#Ensure camera info is correct for projection matrix
#Remember that the projection matrix is:
"""
[f 0 0 0]
[0 f 0 0]
[0 0 alpha beta]
[0 0 -1 0]
"""
z_near = cam.nearPlane
z_far = cam.farPlane

alpha = (z_far + z_near) / (z_near - z_far)
alpha_diff = cam.projectionMatrix[2][2] - alpha
beta = 2 * z_near * z_far / (z_near - z_far)
beta_diff = cam.projectionMatrix[3][2] - beta

tol = 0.00001
correctCamInfo = ((cam.projectionMatrix[2][3] == -1)
             and (alpha_diff < tol)
             and (beta_diff < tol))
print("Projection matrix matches up? " + str(correctCamInfo))

#Display the matrix information
print("Projection matrix is:", cam.projectionMatrix)
print("Focal length is: {:5f}"
    .format(cam.projectionMatrix[0][0]))
