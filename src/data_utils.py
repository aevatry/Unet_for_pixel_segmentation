import numpy as np
import PIL
import os

def clean_image_folder(base_img_folder:str, label_img_folder:str)->None:
    """
    Remove any images in the input image folder that does not have a label image associated in the label folder

    Args:
        base_img_folder (str): input data directory
        label_img_folder (str): label data directory
    """

    base_img_names = get_images_names(base_img_folder)
    label_img_names = get_images_names(label_img_folder)

    # only need to check the input
    for base_name in base_img_names:
        if not(base_name in label_img_names):
            full_path = os.path.abspath(os.path.join(base_img_folder, base_name))
            if os.path.exists(full_path + ".png"):
                os.remove(full_path.join(".png"))
            elif os.path.exists(full_path + ".jpg"):
                os.remove(full_path + ".jpg")
            else:
                raise FileExistsError(f"the file {full_path} is not a recognized image format")
            
            print(f"file {base_name} deleted")


def get_images_names(img_folder:str)->list:

    img_names = []
    for filename in os.listdir(img_folder):
        if filename.endswith((".png", ".jpg")):
            filename = filename.split(sep=".")[0]
            img_names+=[filename]

    return img_names

if __name__=="__main__":
    clean_image_folder("data/JPEGImages", "data/SegmentationClass")