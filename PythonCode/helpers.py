"""A set of functions for helping, such as printing certain objects"""
import numpy as np

def print_config(config):
    """Prints the contents of a config file"""
    print('Config file contains: ')
    config_dict = [{x, tuple(config.items(x))} for x in config.sections()]
    for subdict in config_dict:
        print(subdict)

def is_same_image(img1, img2):
    """Returns true if img1 has the same values and size as img2, else False"""
    size_x, size_y = img1.shape[0:2]
    size_x1, size_y1 = img2.shape[0:2]
    if(size_x != size_x1 or size_y != size_y1):
        return False

    #Check all pixel values match up
    for x in range(size_x):
        for y in range(size_y):
            arr1 = img1[x, y]
            arr2 = img2[x, y]
            if np.all(arr1 != arr2):
                print('Images different at: {} {}'.format(x, y))
                print('First image value is', arr1)
                print('Second image value is', arr2)
                return False
        return True


def prompt_user(message, failed=False):
    """prompts a user with a Y/N message"""
    try:
        input_str = input(message + " (Y/N)").casefold()
        if input_str == 'y':
            return True
        if input_str == 'n':
            return False
        print("Please enter the character y or n (case insensitive)")
        return prompt_user(message)
    except ValueError:
        if failed:
            print("Read no input twice, assuming problem, crashing")
            exit(-1)
        print("Read no input, please enter y or n (case insensitive)")
        return prompt_user(message, failed=True)