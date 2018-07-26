from inviwopy.glm import ivec2
from random import randint

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