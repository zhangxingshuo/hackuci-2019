from face_detection import *
import os
import csv

def list_files(dir):                                                                                                  
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                                                                            
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).next()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:                                                                                        
                r.append(subdir + "/" + file)                                                                         
    return r

if __name__ == '__main__':
    with open('lfw_dataset.csv', 'wb') as f:
        writer = csv.writer(f)
        i = 0
        for file in list_files("C:/Users/andy9/Desktop/Homework/hackuci-2019/lfw"):
            smile = determine_smile(file)
            ratio = find_eye_ratio(file)
            writer.writerow([file, smile, ratio])
            print(i)
            i += 1