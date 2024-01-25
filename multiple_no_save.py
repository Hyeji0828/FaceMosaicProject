from deepface import DeepFace
from retinaface import RetinaFace
import cv2
import os
import numpy as np
import utils
import time

save_dir = "./db"
input_fath = "./db/img2.jpg"
file_path = "./db/multi6.jpg"

margin = 50

start_time = time.time()

faces = RetinaFace.detect_faces(file_path)
img = cv2.imread(file_path)
img_marked = img.copy()

verify_box = None
try: 
    obj = DeepFace.verify(input_fath, img, model_name = 'VGG-Face')
    print(obj)
    x = obj['facial_areas']['img2']['x']
    y = obj['facial_areas']['img2']['y']
    w = obj['facial_areas']['img2']['w']
    h = obj['facial_areas']['img2']['h']
    # img_marked = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    verify_box = [x,y, x+w, y+h]

except:
    print("No Face Verified")
    verify_box = None

for face in faces.values():
    x1, y1, x2, y2 = face['facial_area']
    
    if verify_box == None:
        continue

    iou = utils.IoU(face['facial_area'], verify_box)

    if iou >= 0.5:
        img_marked = cv2.rectangle(img_marked, (x1,y1), (x2,y2), (0, 255, 0), 3)
    else:
        # img_marked = cv2.rectangle(img_marked, (x1,y1), (x2,y2), (255, 0, 0), 3)
        img_marked[y1:y2, x1:x2] = cv2.blur(img_marked[y1:y2, x1:x2], (30, 30))

compute_time = time.time() - start_time
fps = 1 / compute_time
print(f"compute time : {compute_time}")
print(f"fps : {fps}")
print(f"src resolution : {img.shape}")

cv2.imshow("result", img_marked)
cv2.waitKey(0)