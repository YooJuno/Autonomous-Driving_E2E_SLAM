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


transformations = T.Compose(
    [T.Lambda(lambda x: (x / 127.5) - 1.0)])




# 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 #
# 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 #
# 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 #

# FLAG_SERIAL = 'DISCONNECTED'
FLAG_SERIAL = 'CONNECTED'

OS_TYPE = 'MAC' 
# OS_TYPE = 'UBUNTU'

# DRIVING_TYPE = 'MANUAL'
DRIVING_TYPE = 'AUTO'

# DRIVE_WITH_SLAM_TYPE = 'WITH'
DRIVE_WITH_SLAM_TYPE = 'WITHOUT'

# 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 #
# 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 #
# 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 # 여기만 건드세요 #







if OS_TYPE == 'UBUNTU':
    camera_num = -1
elif OS_TYPE == 'MAC':
    camera_num = 0

# for multi thread
Boundary = ''
lock = threading.Lock()
sock = 0


# STM32F411RE 연결할지 말지
if FLAG_SERIAL == 'CONNECTED': # Connected to STM32
    ser = juno.serial_connect(OS_TYPE)



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
        

    def run(self):
        key = -1
        global Boundary # 쓰레드 공유변수

        cap = cv2.VideoCapture(camera_num)    
        # cap = cv2.VideoCapture('/home/yoojunho/바탕화면/v1.mp4')
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        cur_angle = 0


        # STM32 연결돼있고 AUTO모드이면 일단 출발
        if FLAG_SERIAL == 'CONNECTED' and DRIVING_TYPE == 'AUTO':
            ser.write(b'w')
            ser.write(b'w')

        while True:
            ret, frame = cap.read()
            
            key = cv2.waitKey(33)

            if (97 <= key <= 122) or (65 <= key <= 90):
                key = chr(key).lower()
                print(key)

            if not ret:
                print("Failed to capture frame.")
                break
            
            # 이미지 인코딩
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
            image_array = image_array[65:-25, :, :]
            crop_img = image_array.copy()
                

            if DRIVING_TYPE == 'AUTO' : 
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
                
                print(diff_angle)
                cur_angle = steering_angle
                cv2.waitKey(33)

                if FLAG_SERIAL == 'CONNECTED' and Boundary == 'IN BOUNDARY':
                    if diff_angle == 0: 
                        continue
                    
                    elif diff_angle > 0: #angle이 오른쪽으로 꺽여야함
                        for i in range(diff_angle) :
                            ser.write(b'd')

                    else : # angle이 왼쪽으로 꺽여야 함
                        for i in range(-diff_angle) :
                            ser.write(b'a')


            elif DRIVING_TYPE == 'MANUAL' : 

                if FLAG_SERIAL == 'CONNECTED':
                    if key == 'w':
                        ser.write(b'w')

                    elif key == 'a':
                        ser.write(b'a')

                    elif key == 's':
                        ser.write(b's')

                    elif key == 'd':
                        ser.write(b'd')

                    elif key == 'x':
                        ser.write(b'x')
                    
            
            if OS_TYPE == 'MAC':
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
            
            if ( 0 < juno_x and juno_x < 2.2) and ( (juno_z > 0.813*juno_x + -0.163-1) and (juno_z <0.813*juno_x + 0.36) ):
                print("between HD and NH")
            
            elif (2.15 < juno_x) and (1.9 < juno_z and juno_z < 3.4):
                print("Corner 1")
            
            elif ( -2.7 < juno_x and juno_x < 2.15) and (3.4 < juno_z and juno_z < 9.1 ):
                print("between HD and grass")
            
            elif (-3.17 < juno_x and juno_x < -2.8) and (9.1 < juno_z and juno_z < 10.2):
                print("in front of ATM")
            
            elif (-2.8 < juno_x and juno_x < -0.35) and (9.1 < juno_z and juno_z < 10.2):
                print("in front of ATM")
            
            else:
                lock.acquire()
                Boundary = 'OUT OF BOUNDARY'
                lock.release()
                print(Boundary)


            # 나갔으면
            if FLAG_SERIAL== 'CONNECTED' and Boundary == 'OUT OF BOUNDARY':
                ser.write(b's')
            


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

    if DRIVE_WITH_SLAM_TYPE == 'WITH':
        # 문자열 받는 쓰레드 시작
        string_thread = StringThread()
        string_thread.start()

