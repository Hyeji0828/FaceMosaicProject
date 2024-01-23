from deepface import DeepFace
from retinaface import RetinaFace
import cv2
import os

save_dir = "./db"
src_name = "img2.jpg"
multi_name = "multi3.jpg"

margin = 50

faces = RetinaFace.detect_faces(multi_name)
img = cv2.imread(multi_name)
img_marked = img.copy()

obj = DeepFace.verify(src_name, img, model_name = 'VGG-Face')
print(obj)
x = obj['facial_areas']['img2']['x']
y = obj['facial_areas']['img2']['y']
w = obj['facial_areas']['img2']['w']
h = obj['facial_areas']['img2']['h']
img_marked = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
cv2.imshow("result", img_marked)
cv2.waitKey(0)

for face in faces.values():
    x1, y1, x2, y2 = face['facial_area']
    img_marked = cv2.rectangle(img_marked, (x1,y1), (x2,y2), (255, 0, 0), 3)

cv2.imshow("result", img_marked)
cv2.waitKey(0)