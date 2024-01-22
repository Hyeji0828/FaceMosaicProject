from deepface import DeepFace
from deepface.commons import functions
import cv2

# load images
img1_path = "img1.jpg"
img2_path = "img2.jpg"
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# face verification
# obj = DeepFace.verify(img1_path, img2_path, model_name = 'VGG-Face')
# print(obj)
# x = obj['facial_areas']['img1']['x']
# y = obj['facial_areas']['img1']['y']
# w = obj['facial_areas']['img1']['w']
# h = obj['facial_areas']['img1']['h']

# img = cv2.rectangle(img1, (x, y), (x + w, y + h), (255, 0, 0), 5)
# cv2.imshow('result', img)
# cv2.waitKey(0)

# face recognition
# df = DeepFace.find(img1_path, db_path = "./db", model_name = 'VGG-Face')
df = DeepFace.find("./db/test2.jpg", db_path = "./db", model_name = 'VGG-Face')
print(df)

for row in df[0].itertuples():
    x = row[2]
    y = row[3]
    w = row[4]
    h = row[5]

    img = cv2.imread(row[1])
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
    cv2.imshow('result', img)
    cv2.waitKey(0)