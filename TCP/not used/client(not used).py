import cv2
import socket
import struct
import pickle

import threading

PORT = 6396
SERVER_ADDRESS = "127.0.0.1"
# SERVER_ADDRESS = "192.168.1.103"

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_ADDRESS, PORT))

cap = cv2.VideoCapture(-1)
# cap = cv2.VideoCapture("/home/yoojunho/바탕화면/v1.mp4")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    
    ret, frame = cap.read()
    # cv2.imshow('test',frame)

    if not ret:
        print("Failed to capture frame.")
        break
    
    encoded_image = cv2.imencode(".jpg", frame)[1].tobytes()

    size = len(encoded_image).to_bytes(4,byteorder='little')
    
    sock.sendall(size)
    sock.sendall(encoded_image)

    data = sock.recv(1024).decode() # 수신 버퍼 크기는 1024바이트로 설정하고, 수신한 데이터는 디코딩하여 문자열로 변환합니다.
    # 수신한 데이터 출력
    print("Received data:", data)

    # Sleep for 1ms
    cv2.waitKey(1)


cap.release()
sock.close()

