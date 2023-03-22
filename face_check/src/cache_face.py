import csv

import cv2
import torchvision

face_class = cv2.CascadeClassifier("../face_classtool_mod/haarcascade_frontalface_alt2.xml")


def cache_face_from_capture(path_name, name, catch_face_num):
    cap = cv2.VideoCapture(0)
    num = 0
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

                # 将当前帧保存为图片


                img_name = '%s%s_%d.jpg' % (path_name, name, num)
                csv_context = '%s_%d.jpg' % (name, num)
                w_csv = open('../person_csv/person_train.csv', 'a', encoding='utf8', newline='')
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                image_res = cv2.resize(image,(224,224)) # vgg16输入为224
                cv2.imwrite(img_name, image_res)
                writer_csv = csv.writer(w_csv)
                writer_csv.writerow([csv_context, '1'])


                num = num + 1
                if num > (catch_face_num):  # 如果超过指定最大保存数量退出循环
                    break

                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 255), 4)

        if num > (catch_face_num):
            break

        cv2.imshow("window_name", frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cache_face_from_capture(path_name='../train_face_image/', name="Adminer_hs", catch_face_num=256)
    # cache_face_from_capture(path_name='../test_face_image/', name="bdd", catch_face_num=32)