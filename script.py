from __future__ import with_statement
import csv
from face_detection import *

def parse_attrs(filename):
    image_attrs = dict()
    with open("dataset/selfie_dataset.txt") as file:
        for line in file:
            raw_data = line.split()
            name = raw_data[0]
            num_data = [float(elem) for elem in raw_data[1:]]
            image_attrs[name] = (num_data[15] > 1, num_data[17] > 1)
    return image_attrs

if __name__ == '__main__':
    attrs = parse_attrs('dataset/selfie_dataset.txt')
    with open('dataset.csv', 'wb') as f:
        writer = csv.writer(f)
        i = 0
        for key in attrs.keys():
            print(i)
            image_name = 'dataset/images/' + key + '.jpg'
            smiling = determine_smile(image_name)
            attrs[key] = list(attrs[key]) + [smiling]
            writer.writerow(attrs[key])
            i += 1