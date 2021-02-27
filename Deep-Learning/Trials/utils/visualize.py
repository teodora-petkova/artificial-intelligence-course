import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cv2

def show_image(image, title, colour_map = "gray"):
    plt.imshow(image, cmap = colour_map)
    plt.title(title)
    plt.axis("off")
    plt.show()
    
def show_barplot_celltypes(df, database_name=""):
    cells = []
    for cell_type, data in df.groupby(df.label):
        tup = (cell_type, len(data))
        cells.append(tup)
        print(tup)

    fig, ax = plt.subplots()
    cell_types = np.array([str(c[0]) for c in cells])
    counts =[c[1] for c in cells]

    plt.barh(cell_types, counts)
    plt.title(f"Cell types by count {database_name}")
    plt.ylabel("Cell types")
    plt.xlabel("Counts")
    plt.show()

def get_image_with_bounding_boxes(image, bounding_boxes, colour=(0, 0, 255)):
    clone = image.copy()
    for bb in bounding_boxes:
        clone = cv2.rectangle(clone,
                              (bb.xmin, bb.ymin),
                              (bb.xmax, bb.ymax),
                              colour,
                              2)
        cv2.putText(clone,
                    bb.label,  
                    (bb.xmin+5, bb.ymin+15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    colour,
                    1)
    return clone

def show_image_with_bounding_boxes(image, image_title, bounding_boxes, colour=(0, 0, 255)):
    fig,ax = plt.subplots(1, figsize=(10, 10))
    clone = get_image_with_bounding_boxes(image, bounding_boxes)
    ax.imshow(clone, cmap = "gray")
    ax.set_title(image_title)
    ax.axis('off')
    plt.show()

def show_images(images, titles, fig_title=None, fig_width = 13, fig_height = 5):
    assert(len(images) == len(titles))

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    fig.suptitle(fig_title, fontsize = 20)

    i = 1
    for image, title in zip(images, titles):
        plt.subplot(1, len(images), i)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.title(title , fontsize = 20)
        plt.axis('off')
        i += 1

    plt.show()