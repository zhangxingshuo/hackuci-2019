import numpy as np 
import cv2 as cv 
import random as rng
import scipy.spatial

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

def find_mouth(img, show=False):
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
            return None

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

    if show:
        cv.imshow('img', copy)

    return final_mouth

def detect_corners(img, rect, max_corners=128, show=False):
    maxCorners = max(max_corners, 1)
    # Parameters for Shi-Tomasi algorithm
    qualityLevel = 0.01
    minDistance = 5
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
    if corners is None:
        return None, None
    # print('Number of corners detected:', corners.shape[0])
    radius = 4
    for i in range(corners.shape[0]):
        cv.circle(copy, (corners[i,0,0], corners[i,0,1]), radius, (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)), cv.FILLED)
    
    # Show what you got
    if show:
        cv.imshow('corners', copy)

    return copy, corners

def fit_smile(img, corners, show=False):
    x = [corner[0][0] for corner in corners]
    y = [corner[0][1] for corner in corners]
    coeffs = np.polyfit(x, y, 2)

    f = lambda x: coeffs[0]*x**2 + coeffs[1]*x + coeffs[2]
    copy = np.copy(img)
    for i in range(0, copy.shape[1]):
        cv.circle(copy, (i, int(f(i))), 1, (255, 255, 255), cv.FILLED)

    if show:
        cv.imshow('line', copy)

    return coeffs[0]

def determine_smile(image_name, show=False):
    img = cv.imread(image_name)
    mouth = find_mouth(img, show=show)
    if mouth is None:
        return None
    mouth_img, corners = detect_corners(img, mouth, show=show)
    if mouth_img is None or corners is None:
        return None
    return fit_smile(mouth_img, corners, show=show)

def find_eye_ratio(image_name, show=False):
    img = cv.imread(image_name)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) < 1:
        print("No faces found :(")
        return None

    face_area = faces[0][2] * faces[0][3]

    copy = np.copy(img)
    for (x,y,w,h) in eyes:
        cv.rectangle(copy,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = copy[y:y+h, x:x+w]

    if show:
        cv.imshow('img', copy)

    eye_ratio_sum = 0
    count = 0
    for eye in eyes:
        _copy, corners = detect_corners(img, eye)
        points = [tuple(corner[0]) for corner in corners]
        if len(points) < 3:
            continue
        try:
            hull = scipy.spatial.ConvexHull(points)
        except:
            continue
        eye_ratio_sum += hull.volume / face_area 
        count += 1
    if count > 0:
        return eye_ratio_sum / len(eyes)
    else:
        return 0