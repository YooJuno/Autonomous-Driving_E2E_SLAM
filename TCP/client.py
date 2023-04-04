#################### CLIENT ####################
#################### CLIENT ####################
#################### CLIENT ####################
#################### CLIENT ####################


import sys
import threading
import cv2
from PIL import Image
import numpy as np

import torch
from torch.autograd import Variable
import torchvision.transforms as T

import base64
from io import BytesIO

import socket
import client_func as juno
import yolo as od

import pandas as pd
import csv

transformations = T.Compose(
    [T.Lambda(lambda x: (x / 127.5) - 1.0)])




# 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 #
# 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 #
# 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 #

# FLAG_SERIAL = 'DISCONNECTED'
FLAG_SERIAL = 'CONNECTED'

# OS_TYPE = 'MAC' 
OS_TYPE = 'UBUNTU'

# DRIVING_TYPE = 'MANUAL'
# DRIVING_TYPE = 'AUTO'
driving_type = 'AUTO'

DRIVE_WITH_SLAM_TYPE = 'WITH'
# DRIVE_WITH_SLAM_TYPE = 'WITHOUT'



# 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 #
# 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 #







if OS_TYPE == 'UBUNTU':
    camera_num = 2
elif OS_TYPE == 'MAC':
    camera_num = 0

# for multi thread
Boundary = ''
lock = threading.Lock()
sock = 0
interval = 10
frame_yolo = 0 



# STM32F411RE 연결할지 말지
if FLAG_SERIAL == 'CONNECTED': # Connected to STM32
    ser = juno.serial_connect(OS_TYPE)


def detect(img):
    # Stop signal
    sign = 0
    
    # Load Yolo
    net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # img = cv2.resize(img, (640, 480))

    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "person" and w>=40:
                sign = 1
                print("person1")
                break
            else:
                sign = 0

    return sign


class YoloThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        
    def run(self):
        global frame_yolo
        cnt = 0
        while(1):
            if cnt % interval == 0 :
                person = detect(frame_yolo)
                print('person')
                cnt = 0
            cnt += 1
        



# 이미지를 보내는 쓰레드
class ImageThread(threading.Thread):
    def __init__(self, model, args):
        threading.Thread.__init__(self)
        global sock
         # 서버에 연결  
        if DRIVE_WITH_SLAM_TYPE == 'WITH':      
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((args.IP, int(args.PORT)))
            self.conn = sock

        self.model = model
        self.cnt = 0
        

    def run(self):
        key = -1
        global Boundary # 쓰레드 공유변수
        global driving_type
        global frame

        cap = cv2.VideoCapture(camera_num)    
        # cap = cv2.VideoCapture('/home/yoojunho/바탕화면/v1.mp4')
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        cur_angle = 0
        csv_angle = 0
        
        # STM32 연결돼있고 AUTO모드이면 일단 출발
        if FLAG_SERIAL == 'CONNECTED' and driving_type == 'AUTO':
            ser.write(b'w')
            ser.write(b'w')


        # speedz flag
        prev_person = 0
        cnt = 0
        while True:
            ret, frame = cap.read()
            frame_yolo = frame
            key = cv2.waitKey(1)  
            cnt = cnt + 1

            if (97 <= key <= 122) or (65 <= key <= 90):
                key = chr(key).lower()
            elif key == 27:
                break;            
            # frame 경로
            self.path = './data/frame' + str(self.cnt) + '.jpg'
            cv2.imwrite(self.path, frame)
            self.cnt += 1

            if not ret:
                print("Failed to capture frame.")
                break
            
            # 이미지 인코딩F
            encoded_image = cv2.imencode(".jpg", frame)[1].tobytes()

            # SLAM 이용하면 서버와 통신해야됨
            if DRIVE_WITH_SLAM_TYPE == 'WITH':
                size = len(encoded_image).to_bytes(4,byteorder='little')
                self.conn.send(size)
                self.conn.send(encoded_image)

            frame_str = base64.b64encode(encoded_image)
            
            image = Image.open(BytesIO(base64.b64decode(frame_str)))
            image = image.resize((320,160))

            image_array = np.array(image.copy())
            image_array = image_array[40:-50, :]
            crop_img = image_array.copy()
                

            if driving_type == 'AUTO' : 
                # 'p' 눌렸을 때 멈추고 driving mode로 변환
                if key == 'p':
                    ser.write(b's')
                    driving_type = 'MANUAL'
                
                # Object Detection Part!!!
                    
                     

                # transform RGB to BGR for cv2
                image_array = image_array[:, :, ::-1]
                image_array = transformations(image_array)
                image_tensor = torch.Tensor(image_array)
                image_tensor = image_tensor.view(1, 3, 70, 320)
                image_tensor = Variable(image_tensor)

                steering_angle = self.model(image_tensor).view(-1).data.numpy()[0] #angle
                steering_angle = steering_angle * 20
                diff_angle = steering_angle - cur_angle
                # diff_angle = diff_angle / 10
                diff_angle = int(diff_angle)

                #steering_angle 값을 quantization을 해야함
                # steering_angle = steering_angle-0.253
                
                # print(diff_angle)
                cur_angle = steering_angle
                
                # angle > 1 일 때도 고려
                if FLAG_SERIAL == 'CONNECTED' and Boundary == 'IN BOUNDARY':
                    if diff_angle > 0: #angle이 오른쪽으로 꺽여야함
                        for i in range(diff_angle) :
                            ser.write(b'd')
                            csv_angle += 0.25
                            if csv_angle >= 1 :
                                csv_angle = 1

                    else : # angle이 왼쪽으로 꺽여야 함
                        for i in range(-diff_angle) :
                            ser.write(b'a')
                            csv_angle -= 0.25
                            if csv_angle <= -1 :
                                csv_angle = -1
                
                # csv 파일 열기/쓰기
                with open('driving_log_all.csv', 'a', newline='') as csv_file:
                    wr = csv.writer(csv_file)
                    wr.writerow([self.path, str(csv_angle)])

            elif driving_type == 'MANUAL' :
                if key == 'r':
                    driving_type = 'AUTO'
                    ser.write(b'w')
                    ser.write(b'w')
                else:
                    if FLAG_SERIAL == 'CONNECTED':
                        if key == 'w':
                            print("W")
                            ser.write(b'w')

                        elif key == 'a':
                            print("A")
                            ser.write(b'a')
                            csv_angle -= 0.25
                            if csv_angle <= -1 :
                                csv_angle = -1

                        elif key == 's':
                            print("S")
                            ser.write(b's')

                        elif key == 'd':
                            print("D")
                            ser.write(b'd')
                            csv_angle += 0.25
                            if csv_angle >= 1 :
                                csv_angle = 1

                        elif key == 'x':
                            print("X")
                            ser.write(b'x')

                    # csv 파일 열기/쓰기
                    with open('driving_log_keyboard.csv', 'a', newline='') as csv_file:
                        wr = csv.writer(csv_file)
                        wr.writerow([self.path, str(csv_angle)])
                    
                    with open('driving_log_all.csv', 'a', newline='') as csv_file:
                        wr = csv.writer(csv_file)
                        wr.writerow([self.path, str(csv_angle)])
                        
            
            if OS_TYPE == 'UBUNTU':
                cv2.imshow("autodrive_crop", crop_img)

        if DRIVE_WITH_SLAM_TYPE == 'WITH':
            # 연결 종료
            self.conn.close()

# 좌표를 받는 쓰레드
class StringThread(threading.Thread):
    def __init__(self):
        global sock
        threading.Thread.__init__(self)
        self.conn = sock
        
    def run(self):
        
        global Boundary # 쓰레드 공유변수
        # main.
        out_cnt = 0
        while True:
            data = self.conn.recv(1024).decode()
            self.conn.recv(1024).decode()
            
            if not data:
                break
            data = data.split(',')

            juno_x = float(data[0])
            juno_z = float(data[1])
            
            print("x : ", juno_x, "\nz : " , juno_z)
            print()

            lock.acquire()
            Boundary = 'IN BOUNDARY' # 범위 안에 있음
            lock.release()
            
            if ((juno_z < 6.105 * juno_x + 0.388) and (juno_z > 6.105 * juno_x - 0.139) and (juno_z < -0.347 * juno_x + 0.571)):
                out_cnt = 0
                print("between HD and NH")
            elif ((juno_z < -0.347 * juno_x + 0.571) and (juno_z > -0.340 * juno_x + 0.444) and (juno_z < 6.37 * juno_x + 9.446)):
                out_cnt = 0
                print("between HD and grass")
            elif ((juno_z < 6.37 * juno_x + 9.446) and (juno_z > 7.52 * juno_x + 10.4) and (juno_z < -0.317 * juno_x + 1.6)) :
                out_cnt = 0
                print("between ATM and grass")
            elif ((juno_z < -0.317 * juno_x + 1.6) and (juno_z > -0.339 * juno_x + 1.473) and (juno_x  > -1.5)):
                out_cnt = 0
                print("between ATM and grass")
            else : out_cnt = out_cnt + 1
            
            if out_cnt > 3:
                lock.acquire()
                Boundary = 'OUT OF BOUNDARY'
                lock.release()
                print(Boundary)


            # 나갔으면
            if FLAG_SERIAL== 'CONNECTED' and Boundary == 'OUT OF BOUNDARY' and driving_type == 'AUTO':
                ser.write(b's')
                #driving_type = 'MANUAL'
            


        # 연결 종료
        self.conn.close()


if __name__ == '__main__':

    args = juno.parsing()

    model = juno.NetworkNvidia()

    if OS_TYPE == 'UBUNTU': # UBUNTU
        try:
            checkpoint = torch.load(
                args.model, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])

        except KeyError:
            checkpoint = torch.load(
                args.model, map_location=lambda storage, loc: storage)
            model = checkpoint['model']

        except RuntimeError:
            print("==> Please check using the same model as the checkpoint")
            import sys
            sys.exit()
    
    elif OS_TYPE == 'MAC':
        checkpoint = torch.load(
            args.model, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])


    # 이미지 보내는 쓰레드 시작
    image_thread = ImageThread(model, args)
    image_thread.start()
    
    # yolo_thread = YoloThread()
    # yolo_thread.start()
    
    if DRIVE_WITH_SLAM_TYPE == 'WITH':
        # 문자열 받는 쓰레드 시작
        string_thread = StringThread()
        string_thread.start()




