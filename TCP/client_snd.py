import cv2
import socket
import struct
import pickle

PORT = 6395
# SERVER_ADDRESS = "127.0.0.1"
SERVER_ADDRESS = "192.168.1.103"


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_ADDRESS, PORT))

cap = cv2.VideoCapture('/home/yoojunho/바탕화면/v1.mp4')
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
    
    # Sleep for 1ms
    cv2.waitKey(1)


cap.release()
sock.close()

