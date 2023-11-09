import cv2
import numpy as np


# extract the face portion in a given image
def extractFace(image, x1, x2, y1, y2): 
    image_array = np.asarray(image, "uint8")

    y_min = min(y1, y2)
    y_max = max(y1, y2)
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    face = image_array[y_min:y_max, x_min:x_max]
    
    # resize the detected face to 224x224: size required for VGGFace input
    try:
        face = cv2.resize(face, (224, 224) )
        face_array = np.asarray(face,  "uint8")
        return face_array
    except:
        return None


import dlib
import matplotlib.pyplot as plt

detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
conf_thres = 0.6

# detect face in a given image
# returning an array of image_array for all faces detected in a image
# input: image as an array
def detectFace(image): 
    image_array = np.asarray(image, "uint8")
    faces_detected = detector(image_array)
    if len(faces_detected) == 0:
        return []
    faces_extracted = []
    
    for face in faces_detected:
        
        conf = face.confidence
        if conf < conf_thres:
            continue 
                
        x1 = face.rect.left()
        y1 = face.rect.bottom()
        x2 = face.rect.right()
        y2 = face.rect.top()

        
        face_array = extractFace(image, x1, x2, y1, y2)
        if face_array is not None: 
            faces_extracted.append(face_array)
            
    return faces_extracted