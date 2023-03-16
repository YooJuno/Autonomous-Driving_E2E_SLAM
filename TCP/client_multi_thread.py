import argparse
import sys
import serial
import time
import threading
import queue
import cv2
from PIL import Image
import numpy as np

import torch
from torch.autograd import Variable
import torchvision.transforms as T

import imageio as iio
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# juno
import socket

model = None
prev_image_array = None

transformations = T.Compose(
    [T.Lambda(lambda x: (x / 127.5) - 1.0)])

import torch.nn as nn
import torch.nn.functional as F


class NetworkNvidia(nn.Module):
    """NVIDIA model used in the paper."""

    def __init__(self):
        """Initialize NVIDIA model.
        NVIDIA model used
            Image normalization to avoid saturation and make gradients work better.
            Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
            Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
            Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
            Drop out (0.5)
            Fully connected: neurons: 100, activation: ELU
            Fully connected: neurons: 50, activation: ELU
            Fully connected: neurons: 10, activation: ELU
            Fully connected: neurons: 1 (output)
        the convolution layers are meant to handle feature engineering.
        the fully connected layer for predicting the steering angle.
        the elu activation function is for taking care of vanishing gradient problem.
        """
        super(NetworkNvidia, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64 * 2 * 33, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, input):
        """Forward pass."""
        input = input.view(input.size(0), 3, 70, 320)
        output = self.conv_layers(input)
        # print(output.shape)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output


def openSerial(port, baudrate=9600, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=1, xonxoff=False, rtscts=False, dsrdtr=False):
    ser = serial.Serial()

    ser.port = port
    ser.baudrate = baudrate
    ser.bytesize = bytesize
    ser.parity = parity
    ser.stopbits = stopbits
    ser.timeout = timeout
    ser.xonxoff = xonxoff
    ser.rtscts = rtscts
    ser.dsrdtr = dsrdtr

    ser.open()
    return ser

def writePort(ser, data):
    print("send : ", data)
    ser.write(data)


ser = serial.Serial(
                    port='/dev/ttyACM0',
                    baudrate=9600,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS,
                    timeout=0
                )

print('PORT : ',ser.portstr) #연결된 포트 확인.
if ser.isOpen() == False :
    ser.open()




shared_var = 0
lock = threading.Lock()

# 이미지를 보내는 쓰레드
class ImageThread(threading.Thread):
    def __init__(self, conn, model):
        threading.Thread.__init__(self)
        self.conn = conn
        self.model = model

    def run(self):
        global shared_var # 쓰레드 공유변수

        # 웹캠 설정
        cap = cv2.VideoCapture(-1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        cur_angle = 0

        ser.write(b'w')
        ser.write(b'w')

        while True:
            # 이미지 읽기
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame.")
                break

            # 이미지 인코딩
            encoded_image = cv2.imencode(".jpg", frame)[1].tobytes()

            size = len(encoded_image).to_bytes(4,byteorder='little')

            # 이미지 크기 전송
            self.conn.send(size)

            # 이미지 데이터 전송
            self.conn.send(encoded_image)







            retval, buffer = cv2.imencode('.jpg', frame)
            frame_str = base64.b64encode(buffer)
            
            image = Image.open(BytesIO(base64.b64decode(frame_str)))
            image = image.resize((320,160))

            image_array = np.array(image.copy())
            cv2.imshow("autodrive", image_array)
            image_array = image_array[65:-25, :, :]
            cv2.imshow("autodrive_crop", image_array)

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
            if shared_var == 0 :
                if diff_angle == 0: 
                    continue
                
                elif diff_angle > 0: #angle이 오른쪽으로 꺽여야함
                    for i in range(diff_angle) :
                        ser.write(b'd')

                else : # angle이 왼쪽으로 꺽여야 함
                    for i in range(-diff_angle) :
                        ser.write(b'a')



        # 연결 종료
        self.conn.close()

# 문자열을 받는 쓰레드
class StringThread(threading.Thread):
    def __init__(self, conn):
        threading.Thread.__init__(self)
        self.conn = conn

    def run(self):
        
        global shared_var # 쓰레드 공유변수
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
            shared_var = 0
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
                print("Out of boundary")
                lock.acquire()
                shared_var = 1
                lock.release()

            if shared_var == 1:
                ser.write(b's')



        # 연결 종료
        self.conn.close()

# 클라이언트 실행
def client(_model):
    # 서버 IP 주소와 포트 설정
    HOST = '127.0.0.1'
    PORT = 6396

    # 서버에 연결
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))

    # 이미지 보내는 쓰레드 시작
    image_thread = ImageThread(s, _model)
    image_thread.start()

    # 문자열 받는 쓰레드 시작
    string_thread = StringThread(s)
    string_thread.start()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Auto Driving')
    parser.add_argument(
        '--model',
        type=str,
        default='../checkpoint/model-a-100_1.h5',
        help='Path to model h5 file. Model should be on the same path.'
    )
    args = parser.parse_args()

    model = NetworkNvidia()

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

    # 포트 설정
    PORT = '/dev/ttyACM0'
    # 연결




    client(model)
