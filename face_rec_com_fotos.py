import os

import cv2
import face_recognition
import numpy as np

images = "images"

imgTiago = face_recognition.load_image_file(f"{images}/tiago.png")
imgTiago = cv2.cvtColor(imgTiago, cv2.COLOR_BGR2RGB)

imgTiago2 = face_recognition.load_image_file(f"{images}/tiagoRosto.png")
imgTiago2 = cv2.cvtColor(imgTiago2, cv2.COLOR_BGR2RGB)

imgTiagoMarques = face_recognition.load_image_file(f"{images}/tiagomarques.png")
imgTiagoMarques = cv2.cvtColor(imgTiagoMarques, cv2.COLOR_BGR2RGB)


faceLocTiago = face_recognition.face_locations(imgTiago)[0]
encodeTiago = face_recognition.face_encodings(imgTiago)[0]
cv2.rectangle(
    imgTiago,
    (faceLocTiago[3], faceLocTiago[0]),
    (faceLocTiago[1], faceLocTiago[2]),
    (0, 0, 255),
    2,
)

faceLocTiago2 = face_recognition.face_locations(imgTiago2)[0]
encodeTiago2 = face_recognition.face_encodings(imgTiago2)[0]
cv2.rectangle(
    imgTiago2,
    (faceLocTiago2[3], faceLocTiago2[0]),
    (faceLocTiago2[1], faceLocTiago2[2]),
    (0, 0, 255),
    2,
)

faceLocTiagoMarques = face_recognition.face_locations(imgTiagoMarques)[0]
encodeTiagoMarques = face_recognition.face_encodings(imgTiagoMarques)[0]
cv2.rectangle(
    imgTiagoMarques,
    (faceLocTiagoMarques[3], faceLocTiagoMarques[0]),
    (faceLocTiagoMarques[1], faceLocTiagoMarques[2]),
    (0, 0, 255),
    2,
)


resultsTiagoTiago = face_recognition.compare_faces([encodeTiago], encodeTiago2)
distanceTiagoTiago = face_recognition.face_distance([encodeTiago], encodeTiago2)

resultsTiagoTiagoMarques = face_recognition.compare_faces(
    [encodeTiago], encodeTiagoMarques
)
distanceTiagoTiagoMarques = face_recognition.face_distance(
    [encodeTiago], encodeTiagoMarques
)


print("EncodeTiago", encodeTiago)
print("EncodeTiagoMarques", encodeTiagoMarques)

print("Tiago com Tiago", resultsTiagoTiago)
print("Tiago com Tiago", distanceTiagoTiago)

print("Tiago com TiagoMarques", resultsTiagoTiagoMarques)
print("Tiago com TiagoMarques", distanceTiagoTiagoMarques)

cv2.imshow("Tiago 1", imgTiago)
cv2.imshow("Tiago 2", imgTiago2)


cv2.waitKey(0)
