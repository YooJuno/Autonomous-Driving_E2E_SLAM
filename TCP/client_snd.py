import cv2
import socket
import struct

PORT = 8485
# SERVER_ADDRESS = "127.0.0.1"
SERVER_ADDRESS = "192.168.1.103"


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_ADDRESS, PORT))

cap = cv2.VideoCapture(-1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)



print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    
    ret, frame = cap.read()
    print(frame.shape[1], frame.shape[0])
    # cv2.imshow('test',frame)
    frame[0]
    if not ret:
        print("Failed to capture frame.")
        break
    
    _, buffer = cv2.imencode(".jpg", frame)
    size = struct.pack(">L", len(buffer))
    
    sock.sendall(size)
    sock.sendall(buffer)
    
    # Sleep for 1ms
    cv2.waitKey(1)


cap.release()
sock.close()

