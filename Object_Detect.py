import datetime
import cv2
import imutils

PATH_VIDEO = 'video_demo.mp4'

capture = cv2.VideoCapture(PATH_VIDEO)

idx = 0
first_frame = None

# Vẽ vùng quan sát
top_left, bottom_right = (80, 80), (380, 180)
while True:
    ret, frame = capture.read()
    if not ret:
        break
    
    text = 'An toan'
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Nếu khung hình đầu tiên chưa được xác định
    if first_frame is None:
        first_frame = frame
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)
    
    # So sánh khung hình hiện tại với khung hình đầu tiên
    frame_delta = cv2.absdiff(first_gray, gray)
    
    # Chuyển ảnh thành ảnh nhị phân
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    
    # Làm mờ ảnh nhị phân
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Tìm các đối tượng trong ảnh nhị phân
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    
    # Xử lý cho cả 2 phiên bản của OpenCV
    cnts = imutils.grab_contours(cnts)
    
    # Vẽ vùng quan sát
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    for c in cnts:
        # Bỏ qua các đối tượng có diện tích nhỏ
        if cv2.contourArea(c) < 500:
            continue
        
        # Lấy tọa độ của hình chữ nhật bao quanh đối tượng
        (x, y, w, h) = cv2.boundingRect(c)
        # Xác định tâm của đối tượng
        center_x = x + w / 2
        center_y = y + h / 2
        # Kiểm tra đối tượng có nằm trong khu vực quan sát hay không
        logic = top_left[0] < center_x < bottom_right[0] and top_left[1] < center_y < bottom_right[1]
        if logic:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Co xam nhap"
            # Hiện cảnh báo lên hình
            cv2.putText(frame, "Tinh trang: {}".format(text), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imshow("Camera an ninh", frame)
    cv2.imwrite('../images/{}_result.jpg'.format(idx), frame)
    idx += 1
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    
print('Done :)')
cv2.destroyAllWindows()


        
        