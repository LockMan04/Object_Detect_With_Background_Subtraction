"""
Phát hiện đối tượng xâm nhập trong video (Camera hoặc Video)
1. Sử dụng Background Subtractor MOG2 để phân đoạn nền
2. Xử lý ảnh để tăng độ chính xác
3. Tìm các contours trong foreground mask
4. Vẽ bounding box cho các contours
5. Hiển thị kết quả
"""
import cv2
import imutils
import datetime

# Khởi tạo bộ phân đoạn nền MOG2
backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)

# Vùng quan sát
top_left, bottom_right = (0, 0), (380, 280)
idx = 0

# Khởi tạo camera
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    text = 'An Toàn'
    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Phân đoạn nền
    fgMask = backSub.apply(frame)
    thresh = cv2.threshold(fgMask, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    intrusion_detected = False

    # Vẽ bounding box cho các contours
    for c in cnts:
        if cv2.contourArea(c) < 500:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        center_x = x + w / 2
        center_y = y + h / 2

        if top_left[0] < center_x < bottom_right[0] and top_left[1] < center_y < bottom_right[1]:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Có xâm nhập"
            intrusion_detected = True

    # Hiện trạng thái và thời gian
    cv2.putText(frame, f"Tinh trang: {text}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Hiển thị khung hình
    cv2.imshow("Camera an ninh", frame)

    # Lưu ảnh nếu phát hiện xâm nhập
    if intrusion_detected:
        cv2.imwrite(f'../images/{idx}_result.jpg', frame)
        idx += 1

    # Thoát khi nhấn 'q' hoặc ESC
    keyboard = cv2.waitKey(30)
    if keyboard == ord('q') or keyboard == 27:
        break

# Giải phóng tài nguyên
capture.release()
cv2.destroyAllWindows()


