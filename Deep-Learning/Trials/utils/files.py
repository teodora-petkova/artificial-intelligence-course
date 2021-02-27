import os
import numpy as  np
import cv2

def read_image(filepath, cvflags = cv2.IMREAD_ANYCOLOR):
    image = []
    if(os.path.isfile(filepath)):
        try:
            image = cv2.imread(filepath, flags = cvflags)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f'Error reading image {filepath}')
            raise e
    return np.array(image)

def read_text(filepath):
    content = None
    if(os.path.isfile(filepath)):
        with open(filepath) as f:
            content = f.read()
    return content

def make_dir(dir_path, dir_name = ''):
    new_dir_path = os.path.abspath(os.path.join(dir_path, dir_name))

    if not os.path.exists(new_dir_path):
        try:
            os.makedirs(new_dir_path)
        except Exception as e:
            # Raise if directory can't be made, because image cuts won't be saved.
            print('Error creating directory')
            raise e
    else:
        print(f'Directory \'{new_dir_path}\' already exists!')
