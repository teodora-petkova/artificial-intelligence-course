import numpy as np
import xml.etree.ElementTree as etree

class BoundingBox:
    
    def __init__(self, xmin, xmax, ymin, ymax, label):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.label = label

def find_bounding_boxes(xml_as_string):
    bbs = np.array([])
    
    xml = etree.fromstring(xml_as_string)
    if(not(xml is None)):   
        
        objects = [] 
        for i, obj in enumerate(xml.iter('object')):
        
            object_label = obj.find('name').text 
            object_bounding_box = obj.find('bndbox')

            bbs = np.append(bbs, 
                              BoundingBox(int(object_bounding_box.find('xmin').text),
                                          int(object_bounding_box.find('xmax').text),
                                          int(object_bounding_box.find('ymin').text),
                                          int(object_bounding_box.find('ymax').text),
                                          object_label))

    return bbs
