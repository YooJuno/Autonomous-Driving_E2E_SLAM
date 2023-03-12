// 컴파일 명령어
// g++ -o client client_snd.cpp $(pkg-config opencv4 --libs --cflags)


#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

// #define PORT 8485

using namespace std;
using namespace cv;



int main(int argc, char** argv) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        cout << "Failed to create socket." << endl;
        return 1;
    }

    sockaddr_in server;
    server.sin_family = AF_INET;
    // server.sin_addr.s_addr = inet_addr("127.0.0.1"); // 자신의 서버 주소
    server.sin_addr.s_addr = inet_addr(argv[1]); // 외부 서버 주소
    server.sin_port = htons(atoi(argv[2]));

    if (connect(sockfd, (sockaddr*)&server, sizeof(server)) == -1) {
        cout << "Failed to connect to server." << endl;
        return 1;
    }


    // VideoCapture cap(-1);
    VideoCapture cap("/home/yoojunho/바탕화면/v1.mp4");
    cap.set(CAP_PROP_FRAME_WIDTH, 1024);
    cap.set(CAP_PROP_FRAME_HEIGHT, 768);
    cout << cap.get(CAP_PROP_FPS) << endl;
    if (!cap.isOpened()) {
        cout << "Failed to open camera." << endl;
        return 1;
    }

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cout << "Failed to capture frame." << endl;
            break;
        }

        vector<uchar> buffer;
        imencode(".jpg", frame, buffer);

        uint32_t size = buffer.size();
        if (send(sockfd, &size, sizeof(size), 0) != sizeof(size)) {
            cout << "Failed to send size." << endl;
            break;
        }

        if (send(sockfd, buffer.data(), size, 0) != size) {
            cout << "Failed to send data." << endl;
            break;
        }

        // 이미지를 전송한 후, 서버로부터 메시지 수신
        vector<char> buf(100);
        int len = recv(sockfd, buf.data(), buf.size(), 0);
        if (len > 0) {
            buf.resize(len);
            string msg(buf.begin(), buf.end());
            cout << "Received message from server: " << msg << endl;
        } 
        else if (len == 0) {
            cout << "Server disconnected." << endl;
        } 
        else {
            cout << "Failed to receive message from server." << endl;
        }


        // usleep(1000); // 1ms 대기
    }

    cap.release();
    close(sockfd);

    return 0;
}
