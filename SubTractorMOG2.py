import cv2
import numpy as np
import imutils

# Khởi tạo bộ phân đoạn nền MOG2
backSub = cv2.createBackgroundSubtractorMOG2(varThreshold = 16, detectShadows = False)

PATH_VIDEO = 'video_demo.mp4'

capture = cv2.VideoCapture(PATH_VIDEO)

while True:
    _, frame = capture.read()
    if not _:
        break
    
    frame = imutils.resize(frame, width=700)
    # Dùng background subtractor để phân đoạn nền (Foreground Mask - fgMask)
    fgMask = backSub.apply(frame)
    fgMask = cv2.cvtColor(fgMask, 0)

    # Xử lý ảnh để tăng độ chính xác
    kernel = np.ones((5, 5), np.uint8)
    fgMask = cv2.erode(fgMask, kernel, iterations=1)
    fgMask = cv2.dilate(fgMask, kernel, iterations=1)
    fgMask = cv2.GaussianBlur(fgMask, (3, 3), 0)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
    _, fgMask = cv2.threshold(fgMask, 130, 255, cv2.THRESH_BINARY)

    # Tìm các contours trong fgMask
    fgMask = cv2.Canny(fgMask, 20, 200)
    contours, _ = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Vẽ bounding box cho các contours
    for i in range(len(contours)):
        (x, y, w, h) = cv2.boundingRect(contours[i])
        area = cv2.contourArea(contours[i])
        if area > 300:
            cv2.drawContours(fgMask, contours[i], 0, (0, 0, 255), 6)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    
cv2.destroyAllWindows()
