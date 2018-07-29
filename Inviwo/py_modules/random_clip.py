from inviwopy.glm import ivec2
from random import randint, random

def random_subset(network, type):
    """type is expected to be "X", "Y" or "Z" """
    if type is "X":
        clip = network.VolumeSubset.rangeX
    if type is "Y":
        clip = network.VolumeSubset.rangeY
    if type is "Z":
        clip = network.VolumeSubset.rangeZ
    max = (clip.rangeMax // 2) - (clip.rangeMax // 10)
    start = randint(0, max)
    end = randint(0, max)
    end = clip.rangeMax - end
    clip_range = ivec2(start, end)
    clip.value = clip_range

def restore_volume(network, type):
    """type is expected to be "X", "Y" or "Z" """
    if type is "X":
        clip = network.VolumeSubset.rangeX
    if type is "Y":
        clip = network.VolumeSubset.rangeY
    if type is "Z":
        clip = network.VolumeSubset.rangeZ
    clip.value = ivec2(0, clip.rangeMax)
    
def random_clip(network, type):
    """type is expected to be "X", "Y" or "Z" """
    if type is "X":
        clip = network.CubeProxyGeometry.clipX
    if type is "Y":
        clip = network.CubeProxyGeometry.clipY
    if type is "Z":
        clip = network.CubeProxyGeometry.clipZ
    max = (clip.rangeMax // 2) - (clip.rangeMax // 10)
    start = randint(0, max)
    end = randint(0, max)
    end = clip.rangeMax - end
    clip_range = ivec2(start, end)
    clip.value = clip_range

def restore_clip(network, type):
    """type is expected to be "X", "Y" or "Z" """
    if type is "X":
        clip = network.CubeProxyGeometry.clipX
    if type is "Y":
        clip = network.CubeProxyGeometry.clipY
    if type is "Z":
        clip = network.CubeProxyGeometry.clipZ
    clip.value = ivec2(0, clip.rangeMax)

def random_clip_lf(network, lf):
    """Randomly cips a volume for a given lf camera"""
    look_from = lf.look_from
    max_val, max_type = -1, None

    val = abs(look_from.x)
    if val > max_val:
        max_val, max_type = val, "X"

    val = abs(look_from.y)
    if val > max_val:
        max_val, max_type = val, "Y"

    val = abs(look_from.z)
    if val > max_val:
        max_val, max_type = val, "Z"

    random_clip(network, max_type)

    return max_val, max_type

def random_plane_clip(network, lf):
    mesh_clip = network.MeshClipping
    cam = network.EntryExitPoints.camera
    cam.setLook(lf.look_from, lf.look_to, lf.look_up)
    mesh_clip.alignPlaneNormalToCameraNormal.press()
    mesh_clip.getPropertyByIdentifier("movePointAlongNormal").value = True
    rand_clip = random() / 1.4
    mesh_clip.getPropertyByIdentifier("pointPlaneMove").value = rand_clip