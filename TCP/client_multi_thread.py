import socket
import threading
import cv2
import numpy as np

shared_var = 0
lock = threading.Lock()

# 이미지를 보내는 쓰레드
class ImageThread(threading.Thread):
    def __init__(self, conn):
        threading.Thread.__init__(self)
        self.conn = conn

    def run(self):
        # 웹캠 설정
        cap = cv2.VideoCapture(-1)
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


        # 연결 종료
        self.conn.close()

# 문자열을 받는 쓰레드
class StringThread(threading.Thread):
    def __init__(self, conn):
        threading.Thread.__init__(self)
        self.conn = conn

    def run(self):
        
        # main.
        while True:
            data = self.conn.recv(1024).decode()
            self.conn.recv(1024).decode()
            
            if not data:
                break
            data = data.split(',')

            juno_x = float(data[0])
            juno_z = float(data[1])
            # 문자열 출력
            print("x : ", juno_x, "\nz : " , juno_z)
            print()
            
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
            


        # 연결 종료
        self.conn.close()

# 클라이언트 실행
def client():
    # 서버 IP 주소와 포트 설정
    HOST = '127.0.0.1'
    PORT = 6396

    # 서버에 연결
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))

    # 이미지 보내는 쓰레드 시작
    image_thread = ImageThread(s)
    image_thread.start()

    # 문자열 받는 쓰레드 시작
    string_thread = StringThread(s)
    string_thread.start()

if __name__ == '__main__':
    client()
