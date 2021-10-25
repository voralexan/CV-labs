import cv2
import numpy as np

if __name__ == "__main__":
    # Load the cascade for face recognition and read an image
   face_cascade = cv2.CascadeClassifier('trained_face_recognition.xml')
   img = cv2.imread("image.png", 1)
   ###########################
   # 1. Найти лицо на изображении.
   ###########################
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(gray, 1.1, 1)

   ###########################
   # 2. Отступить на 10% от границ лица
   ###########################
   for (x, y, w, h) in faces:
       crop_img = img[int((y-y*0.1)):int(y+h*1.1), int((x-x*0.1)):int(x+w*1.1)].copy()
   for (x, y, w, h) in faces:
       cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

   ###########################
   # 3. Получить бинарное изображение краев (границ объектов).
   ###########################
   img_binary = cv2.Canny(crop_img,150,150)

   ###########################
   # 4. Удалить мелкие границы у которых длина и ширина меньше 10.
   ###########################
   cropped_img_grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
   nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(img_binary, 4)
   mask = np.zeros(cropped_img_grey.shape, dtype="uint8")

   for i in range(1, nb_components):
      w = stats[i, cv2.CC_STAT_WIDTH]
      h = stats[i, cv2.CC_STAT_HEIGHT]

      if w < 10 and h < 10:
         continue

      componentMask = (labels == i).astype("uint8") * 255
      mask = cv2.bitwise_or(mask, componentMask)

   cv2.imshow("Edges", mask)

   ###########################
   # 5. Применить морфологическую операцию наращивания (размер структурирующего элемента 5 x 5).
   ###########################
   kernel = np.ones((5, 5), 'uint8')
   img_dilated = cv2.dilate(mask, kernel, iterations=1)

   ###########################
   # 6. Сгладить полученное изображение краев гауссовским фильтром 5 на 5.
   ###########################
   kernel = np.ones((5,5),np.float32)/25
   img_M = cv2.filter2D(img_dilated,-1,kernel)
   img_M_bw = img_M.copy()
   for x in range(0,img_M.shape[0]):
      for y in range(0,img_M.shape[1]):
         if img_M[x,y] > 127:
            img_M_bw[x,y] = 1
         else:
            img_M_bw[x,y] = 0
   ###########################
   # 7. Получить изображение F1 лица с примененной билатеральной фильтрацией.
   ###########################
   img_F1 = cv2.bilateralFilter(crop_img, 5, 50, 100)

   ###########################
   # 8. Получить изображение F2 лица с улучшенной четкостью/контрастностью.
   ###########################
   img_F2=cv2.addWeighted(crop_img,1.2,np.zeros(crop_img.shape, crop_img.dtype),0,0)

   ###########################
   # 9. Осуществить финальную фильтрацию по формуле
   ###########################
   img_result = crop_img.copy()
   img_result.setflags(write=True)
   for x in range(0,crop_img.shape[0]):
      for y in range(0,crop_img.shape[1]):
         for c in range(0,2):
            img_result[x,y,c] = img_M_bw[x,y]*img_F2[x,y,c]+(1-img_M_bw[x,y])*img_F1[x,y,c]

   cv2.imshow('img', img)
   cv2.imshow('cropimg', crop_img)
   cv2.imshow('binary', img_binary)
   cv2.imshow('dilation', img_dilated)
   cv2.imshow('M', img_M)
   cv2.imshow('F1', img_F1)
   cv2.imshow('F2', img_F2)
   cv2.imshow('result', img_result)
   cv2.waitKey()