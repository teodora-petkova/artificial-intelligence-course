import numpy as np
import matplotlib.pyplot as plt
import time
import random as rng
import requests
import os
import xml.etree.ElementTree as et

import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix

def read_image_by_url(url):
    """
    read an image by a given url containing an image
    """
    response = requests.get(url, stream = True).raw
    image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def read_text_by_url(url):
    """
    read a text file by a given url containing a text
    """
    response = requests.get(url, stream = True)
    return response.content

def read_images_by_dir(dir, ):
    images = np.array(np.empty(64 * 64))

    for filepath in sorted(os.listdir(dir)):
        print(filepath)
        image_filepath = os.path.join(dir, filepath)
        if(os.path.isfile(image_filepath) and image_filepath.endswith('.png')):
            image = cv2.imread(image_filepath, flags = cv2.IMREAD_GRAYSCALE)
            if(image is not None):
                images = np.vstack((images, image.flatten()))
      
    images = images[1:] # remove the first empty element
    return images

def read_labels_by_filepath(filepath):
    labels = np.array([])
    labels_file = open(filepath,"r")
    while True: 
        line_with_label = labels_file.readline()
        if not line_with_label: 
            break
        labels = np.append(labels, line_with_label.rstrip('\n'))
    labels_file.close ()
  
    return labels

def get_image_shape(image_path):
    image = read_image_by_url(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_shape = gray_image.shape
    return gray_image, image_shape

def find_bounding_box(xml_as_string):
    labels = set()
    items = np.array([])
    
    xml = et.fromstring(xml_as_string)
    if(not(xml is None)):   
        
        image_name, _ = os.path.splitext(xml.find('filename').text)
        
        for i, obj in enumerate(xml.iter('object')):
            object_number = str(i + 1).zfill(5) 
            object_name = f"{image_name}_{object_number}.png"
            object_label = obj.find('name').text 
            object_bounding_box = obj.find('bndbox')
            labels.add(object_label)

            items = np.append(items, {
                'object_number': object_number,
                'path': os.path.join(image_name), 
                'name': object_name, 
                'xmin': object_bounding_box.find('xmin').text,
                'xmax': object_bounding_box.find('xmax').text, 
                'ymin': object_bounding_box.find('ymin').text, 
                'ymax': object_bounding_box.find('ymax').text, 
                'label': object_label
            })

    return { 'items': items, 'labels': labels }

def make_dir(path, name = ''):
    path = os.path.abspath(os.path.join(path, name))

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            # Raise if directory can't be made, because image cuts won't be saved.
            print('Error creating directory')
            raise e

def resize_with_padding(image, desired_size, border_type = cv2.BORDER_REPLICATE):
    # actual resize to the closest size to our desired one
    old_width, old_height = image.shape
    ratio = float(desired_size) / max(old_width, old_height)
    new_width, new_height = int(old_width * ratio), int(old_height * ratio)
    image = cv2.resize(image, (new_width, new_height))

    # padding
    delta_width = desired_size - new_width
    delta_height = desired_size - new_height

    top = delta_height//2
    bottom = delta_height - top
    left = delta_width//2
    right = delta_width - left
 
    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, border_type, value=color)
    return new_image

def increase_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
    l2 = clahe.apply(l)

    lab = cv2.merge((l2,a,b))
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) 
    
    return img2
    
def preprocess(image):
    #contrasted_image = increase_contrast(image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    smoothen_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    otsu_threshold, otsu_image = cv2.threshold(smoothen_image, 0, 255,
                                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    image_closed_otsu = cv2.morphologyEx(otsu_image, cv2.MORPH_CLOSE, 
                                        structuring_element, iterations = 3)
    
    canny_image = cv2.Canny(image_closed_otsu, otsu_threshold, 0.5*otsu_threshold)
    return canny_image

def get_cell_image_by_bounding_box(image, bounding_box, dimension):
    xmin, xmax = int(bounding_box.get('xmin')), int(bounding_box.get('xmax'))
    ymin, ymax = int(bounding_box.get('ymin')), int(bounding_box.get('ymax'))

    # check for incorrect annotations of the bounding box
    if(ymax - ymin > 0 and xmax - xmin > 0):
        cropped_image = image[ymin:ymax, xmin:xmax]
        resized_image = cropped_image
        #processed_image = preprocess(cropped_image)
        processed_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
        resized_image = resize_with_padding(processed_image, dimension)
        return cropped_image, resized_image
    else: 
        return None, None

def save_image(image, path):
    try:
        cv2.imwrite(path, image)
    except Exception as  e:
        print(f"Error saving image: {path}")
        print(f"ERROR: {str(e)}")

def show_images(images, shape, fig_width = 13, fig_height = 5):
    plt.figure(figsize=(13, 5))
    i = 1
    for image in images:
        plt.subplot(1, len(images), i)
        plt.imshow(image.reshape(shape), cmap=plt.cm.gray)
        plt.title('Training: %i\n' % i, fontsize = 20)
        plt.axis('off')
        i += 1
    plt.show()

def show_scores(estimator, X_train, y_train, X_test, y_test):
        print("Train: ", estimator.score(X_train, y_train))
        print("Test: ", estimator.score(X_test, y_test)) 

def create_image_samples(cells_dir, labels_filepath):
    URL = "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/"
    min_index, max_index = 0, 10
    dimension = 64

    original_cells_dir = os.path.join(cells_dir, "original")
    make_dir(original_cells_dir)
    
    images = np.array(np.empty(dimension * dimension))
    labels = np.array([])

    for i in range(min_index, max_index):
        print(i)
        # images & annotations paths
        image_file_name = f"BloodImage_{str(i).zfill(5)}"
        image_path = f"{URL}JPEGImages/{image_file_name}.jpg"
        annotations_path = f"{URL}Annotations/{image_file_name}.xml"

        # read data - images with annotations
        image = read_image_by_url(image_path)
        text = read_text_by_url(annotations_path)

        requests.get(image_path, stream = True)
        if (image is not None):
            
            items = find_bounding_box(text).get('items')

            if(len(items) > 0):
                for item in items:
                    cropped_cell, resized_cell = get_cell_image_by_bounding_box(image, item, dimension)

                    if(cropped_cell is not None):
                        save_image(resized_cell, os.path.join(cells_dir, item.get('name')))
                        save_image(cropped_cell, os.path.join(original_cells_dir, item.get('name')))

                        images = np.vstack((images, resized_cell.flatten()))
                        labels = np.append(labels, item.get('label'))

                        labels_file = open(labels_filepath, "w")
                        for label in labels:
                            labels_file.write(f"{label}\n")
                        labels_file.close()
  
    images = images[1:] # remove the first empty element
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    return images, labels

"""
print(np.any(np.isnan(images[0])))
np.savetxt('test2.out', images[0].reshape(100, 100), delimiter=',')
plt.imshow(images[0].reshape(100, 100), cmap=plt.cm.gray)
plt.title("Training: 1")
plt.axis('off')
plt.show()
"""

start_time = time.time()

cells_dir = os.path.join(".", "data", "cells")
make_dir(cells_dir)
labels_file = os.path.join(cells_dir, "labels.txt")

images, labels = create_image_samples(cells_dir, labels_file)
#images = read_images_by_dir(cells_dir)
#labels = read_labels_by_filepath(labels_file)
#assert(np.array_equiv(images, images2))
#assert(np.array_equiv(labels, labels2))

images_pca = PCA(n_components=64)
pca_fit = images_pca.fit(images)
transformed_images = pca_fit.transform(images) 
new_shape = (8, 8)
show_images(images[0:6], (64, 64))
show_images(transformed_images[0:6], new_shape)

plt.plot(np.cumsum(pca_fit.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

reversed_images = pca_fit.inverse_transform(transformed_images)
show_images(reversed_images[0:6], (64, 64))

# Generate data
X_train, X_test, y_train, y_test = train_test_split(transformed_images, labels, test_size=0.2, stratify=labels)
# Initialize SVM classifier
clf = svm.SVC(kernel='linear')

# Fit datam
clf = clf.fit(X_train, y_train)

show_scores(clf, X_train, y_train, X_test, y_test)

matrix = plot_confusion_matrix(clf, 
                               X_test, 
                               y_test,
                               cmap=plt.cm.Greens,
                               normalize='true')
plt.title('Confusion matrix for our classifier')
plt.show(matrix)
plt.show()

# Get support vectors
support_vectors = clf.support_vectors_

# Visualize support vectors
rbcs = X_train[y_train == 'RBC']
wbcs = X_train[y_train == 'WBC']
platelets = X_train[y_train == 'Platelets']
print(rbcs.shape)
print(wbcs.shape)
print(platelets.shape)
print(X_train.shape)

end_time = time.time()
elapsed_time = end_time - start_time
print("Total elapsed time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))