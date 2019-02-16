from __future__ import with_statement
import numpy as np 
import cv2 as cv 

def parse_attrs(filename):
    image_attrs = dict()
    with open("dataset/selfie_dataset.txt") as file:
        for line in file:
            raw_data = line.split()
            name = raw_data[0]
            num_data = [float(elem) for elem in raw_data[1:]]
            image_attrs[name] = num_data
    return image_attrs

def overlap(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    hoverlaps = True
    voverlaps = True
    if (x1 > x2 + w2) or (x1 + w1 < x2):
        hoverlaps = False
    if (y1 > y2 + h2) or (y1 + h1 < y2):
        voverlaps = False
    return hoverlaps and voverlaps

def overlap_cascades(l1, l2):
    overlaps = []
    for elem1 in l1:
        for elem2 in l2:
            if overlap(elem1, elem2):
                overlaps.append(elem1)
    return overlaps

def find_mouth(image_name):
    img = cv.imread(image_name)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    smile_cascade = cv.CascadeClassifier('haarcascade_smile.xml')
    mouth_cascade = cv.CascadeClassifier('mouth.xml')
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    smiles = smile_cascade.detectMultiScale(gray, 1.3, 5)
    mouths = mouth_cascade.detectMultiScale(gray, 1.3, 5)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = [(x, y + int(2 *(1.0 * h / 3)), w, int((1.0 * h) / 3)) for (x, y, w, h) in faces]

    smile_mouth_overlaps = overlap_cascades(smiles, mouths)
    smile_face_overlaps = overlap_cascades(smiles, faces)
    mouth_face_overlaps = overlap_cascades(faces, mouths)
    all_overlaps = overlap_cascades(smile_mouth_overlaps, faces)

    min_area = lambda x: x[2] * x[3]
    final_mouth = None
    if all_overlaps:
        final_mouth = min(all_overlaps, key=min_area)
    elif smile_face_overlaps:
        final_mouth = min(smile_face_overlaps, key=min_area)
    elif mouth_face_overlaps:
        final_mouth = min(mouth_face_overlaps, key=min_area)
    elif smile_mouth_overlaps:
        final_mouth = min(smile_mouth_overlaps, key=min_area)
    else:
        if faces:
            final_mouth = faces[0]
        else:
            print("No face found :(")
            return

    for (x,y,w,h) in smiles:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    for (x,y,w,h) in mouths:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    x, y, w, h = final_mouth
    cv.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    find_mouth('dataset/images/924325_627521150659947_2089248515_a.jpg')