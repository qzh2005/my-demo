import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_face(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return None, img
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])   # 取最大的一张脸
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return gray[y:y + h, x:x + w], img

def calc_similarity(face1, face2):
    f1 = cv2.resize(face1, (100, 100))
    f2 = cv2.resize(face2, (100, 100))
    h1 = cv2.calcHist([f1], [0], None, [256], [0, 256])
    h2 = cv2.calcHist([f2], [0], None, [256], [0, 256])
    cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX)
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

face1, img1 = get_face("1.jpg")
face2, img2 = get_face("2.ls.png")

if face1 is None:
    print("第一张图没有检测到人脸")
elif face2 is None:
    print("第二张图没有检测到人脸")
else:
    sim = calc_similarity(face1, face2)
    print(f"两张人脸相似度：{sim:.4f}")
    label = "Same Person" if sim > 0.7 else "Different People"
    print("判断：" + label)
    cv2.putText(img1, f"Sim:{sim:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img2, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("1.jpg", img1)
    cv2.imshow("2.jpg", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
