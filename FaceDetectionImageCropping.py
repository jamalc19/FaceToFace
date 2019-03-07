import os
import numpy as np
from autocrop import autocrop as ac
from shutil import copy

path = "./Parsed/"

def crop(path = "./Parsed/", dest = "./Cropped/", pixel = 256):
    """
    Crop all of the images in each of the folders in the root directory
    and insert those images into a folder named "crop" in each image folder.

    path (string): Root directory where image folders are located in.
    pixel (int): the output size of one dimension of the cropped images (squares).

    returns None
    """

    # For every person
    for p_id in os.listdir(path):
        in_dir = path + p_id
        crp_dir = dest + p_id
        err_dir = dest + p_id + "\\reject"

        # Make a new folder if it doesn't exist
        if os.path.isdir(crp_dir) == False:
            os.mkdir(crp_dir)
            if os.path.isdir(err_dir) == False:
                os.mkdir(err_dir)

        # Crop all the images in the folder and store them in "crop"
        ac.main(
            input_d=in_dir,
            output_d=crp_dir,
            reject_d=err_dir,
            fheight=pixel,
            fwidth=pixel,
            facePercent=85)

if __name__ == "__main__":
    crop()
