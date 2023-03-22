import sys
sys.path.append("D:\\pythonData\\face_check")
import cv2
import torch
import torchvision


try:
    face_class = cv2.CascadeClassifier("../face_classtool_mod/haarcascade_frontalface_alt2.xml")
finally:
    face_class = cv2.CascadeClassifier("D:\\pythonData\\face_check\\face_classtool_mod\\haarcascade_frontalface_alt2.xml")


#vgg16_mod = torch.load("../trained_mod/vgg16_9.pth", map_location=torch.device("cpu"))

vgg16_mod = torch.load("D:\\pythonData\\face_check\\trained_mod\\vgg16_9.pth", map_location=torch.device("cpu"))

def cache_face_from_capture():
    isnotAdmin_num = 0
    isAdmin_num = 0
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        flag, frame = cap.read()  # 读取一帧数据
        if not flag:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像
        face_Caches = face_class.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=1, minSize=(256, 256))
        if len(face_Caches) > 0:  # 大于0则检测到人脸
            for face_cache in face_Caches:  # 单独框出每一张人脸
                x, y, w, h = face_cache
                # 画出矩形框
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                image_res = cv2.resize(image, (224, 224))  # vgg16输入为224
                transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
                image_totensor = transform(image_res)
                image_input = torch.reshape(image_totensor, (1, 3, 224, 224))
                vgg16_mod.eval()
                with torch.no_grad():
                    image_output = vgg16_mod(image_input)
                isAdmin = image_output.argmax(1).item()
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                if (isAdmin == 0):
                    cv2.putText(frame, '0', (x + 30, y + 30), font, 1, (255, 0, 255), 4)
                    isnotAdmin_num = isnotAdmin_num + 1
                    if (isnotAdmin_num == 15):
                        return 0
                if (isAdmin == 1):
                    cv2.putText(frame, '1', (x + 30, y + 30), font, 1, (255, 0, 255), 4)
                    isAdmin_num = isAdmin_num + 1
                    if(isAdmin_num == 6):
                        return 1

        cv2.imshow("window_name", frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    x = cache_face_from_capture()
    if (x == 1):
        print("1")
    else:
        print("0")
