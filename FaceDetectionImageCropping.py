import os
import numpy as np
from autocrop import autocrop as ac
from shutil import copy


def crop(path="./Parsed3_15/", dest="./Cropped128/", pixel=128):
    """
    Crop all of the images in each of the folders in the root directory
    and insert those images into a folder named "crop" in each image folder.

    path (string): Root directory where image folders are located in.
    pixel (int): the output size of one dimension of the cropped images (squares).

    returns None
    """
    reject_dir ="./reject"
    # Make a reject folder
    if os.path.isdir(reject_dir) == False:
        os.mkdir(reject_dir)

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
            facePercent=90)


def organize(path="./Cropped128/", dest="./OrganizedImages/"):
    '''
    Organizes cropped images into emotion folders.

    returns None
    '''

    # For every person in the root folder
    for p_id in os.listdir(path):
        in_dir = path + p_id
        # For every emotion in a person folder
        for img in os.listdir(in_dir):
            emo = img.split("_")[0]
            new_dir = dest + "\\" + emo
            # Make an emotion folder if it doesn't exist
            if os.path.isdir(new_dir) == False:
                os.mkdir(new_dir)
            # Copy the image to the new destination
            img_path = in_dir + "\\" + img
            copy(img_path, new_dir)


if __name__ == "__main__":
    crop()
    # organize()