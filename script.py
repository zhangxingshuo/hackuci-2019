from __future__ import with_statement
import numpy as np 
import cv2 as cv 
import random as rng

def parse_attrs(filename):
    image_attrs = dict()
    with open("dataset/selfie_dataset.txt") as file:
        for line in file:
            raw_data = line.split()
            name = raw_data[0]
            num_data = [float(elem) for elem in raw_data[1:]]
            image_attrs[name] = (num_data[15] > 1, num_data[17] > 1)
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

def find_mouth(img):
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
            exit(1)

    copy = np.copy(img)
    for (x,y,w,h) in smiles:
        cv.rectangle(copy,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = copy[y:y+h, x:x+w]
    for (x,y,w,h) in mouths:
        cv.rectangle(copy,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = copy[y:y+h, x:x+w]
    for (x,y,w,h) in faces:
        cv.rectangle(copy,(x,y),(x+w,y+h),(255,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = copy[y:y+h, x:x+w]

    x, y, w, h = final_mouth
    cv.rectangle(copy,(x,y),(x+w,y+h),(255,255,255),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = copy[y:y+h, x:x+w]

    # cv.imshow('img', copy)

    return final_mouth

def detect_corners(img, rect, max_corners=128):
    maxCorners = max(max_corners, 1)
    # Parameters for Shi-Tomasi algorithm
    qualityLevel = 0.01
    minDistance = 10
    blockSize = 3
    gradientSize = 3
    useHarrisDetector = True
    k = 0.04

    x, y, w, h = rect
    copy = np.copy(img[y:y+h, x:x+w])
    src_gray = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)

    # Apply corner detection
    corners = cv.goodFeaturesToTrack(src_gray, maxCorners, qualityLevel, minDistance, None, \
        blockSize=blockSize, gradientSize=gradientSize, useHarrisDetector=useHarrisDetector, k=k)
    # Draw corners detected
    print('Number of corners detected:', corners.shape[0])
    radius = 4
    for i in range(corners.shape[0]):
        cv.circle(copy, (corners[i,0,0], corners[i,0,1]), radius, (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)), cv.FILLED)
    
    # Show what you got
    # cv.imshow('corners', copy)

    return copy, corners

def fit_smile(img, corners):
    x = [corner[0][0] for corner in corners]
    y = [corner[0][1] for corner in corners]
    coeffs = np.polyfit(x, y, 2)

    f = lambda x: coeffs[0]*x**2 + coeffs[1]*x + coeffs[2]
    copy = np.copy(img)
    for i in range(0, copy.shape[1]):
        cv.circle(copy, (i, int(f(i))), 1, (255, 255, 255), cv.FILLED)

    # cv.imshow('line', copy)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return coeffs[0] < 0

def determine_smile(image_name):
    img = cv.imread(image_name)
    mouth = find_mouth(img)
    mouth_img, corners = detect_corners(img, mouth)
    return determine_smile(mouth_img, corners)

if __name__ == '__main__':
    attrs = parse_attrs('dataset/selfie_dataset.txt')
    print(attrs['0a7c576672f111e29f1422000a1fbc0e_6'])