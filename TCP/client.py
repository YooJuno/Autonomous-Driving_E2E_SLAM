#################### CLIENT ####################
#################### CLIENT ####################
#################### CLIENT ####################
#################### CLIENT ####################


import argparse
import os , sys
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
import client_func as juno


import torch.nn as nn
import torch.nn.functional as F


transformations = T.Compose(
    [T.Lambda(lambda x: (x / 127.5) - 1.0)])


flag_serial = 0
flag_camera_num = 0
flag_mac_os = 0

# for multi thread
shared_var = 0
lock = threading.Lock()


# STM32F411RE 연결
if flag_serial == 1:
    os.system("sudo chmod 777 /dev/ttyACM0")
    ser = juno.serial_connect(flag_mac_os)


# 이미지를 보내는 쓰레드
class ImageThread(threading.Thread):
    def __init__(self, conn, model):
        threading.Thread.__init__(self)
        self.conn = conn
        self.model = model

    def run(self):
        global shared_var # 쓰레드 공유변수

        cap = cv2.VideoCapture(flag_camera_num)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        cur_angle = 0

        if flag_serial == 1:
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

            frame_str = base64.b64encode(encoded_image)
            
            image = Image.open(BytesIO(base64.b64decode(frame_str)))
            image = image.resize((320,160))

            image_array = np.array(image.copy())
            image_array = image_array[65:-25, :, :]
            
            if flag_mac_os == 0:
                cv2.imshow("autodrive", image_array)
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
            if flag_serial == 1 and shared_var == 0:
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

# 좌표를 받는 쓰레드
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
            shared_var = 0 # 범위 안에 있음
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

            # 나갔으면
            if flag_serial== 1 and shared_var == 1:
                ser.write(b's')

        # 연결 종료
        self.conn.close()


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Auto Driving')
    # parser.add_argument('--model',type=str,default='../model/model-a-100_1.h5',help='')
    # parser.add_argument('--IP',type=str,default='127.0.0.1',help='')
    # parser.add_argument('--PORT',type=str,default='6395',help='')
    # args = parser.parse_args()

    args = juno.parsing()

    model = juno.NetworkNvidia()

    if flag_mac_os == 0:
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
    
    else:
        checkpoint = torch.load(
            args.model, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])


    # 서버에 연결
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((args.IP, int(args.PORT)))

    # 이미지 보내는 쓰레드 시작
    image_thread = ImageThread(s, model)
    image_thread.start()

    # 문자열 받는 쓰레드 시작
    string_thread = StringThread(s)
    string_thread.start()

