// 컴파일 명령어
// g++ -o server server_rcv.cpp $(pkg-config opencv4 --libs --cflags)

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#define PORT 6395

using namespace std;
using namespace cv;



int main(){
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        cout << "Failed to create socket." << endl;
        return 1;
    }
    
    sockaddr_in server;
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port = htons(PORT);
    
    if (bind(sockfd, (sockaddr*)&server, sizeof(server)) == -1) {
        cout << "Failed to bind socket." << endl;
        return 1;
    }
    
    if (listen(sockfd, 10) == -1) {
        cout << "Failed to listen on socket." << endl;
        return 1;
    }
    

    sockaddr_in client;
    socklen_t clientSize = sizeof(client);
    int clientfd = accept(sockfd, (sockaddr*)&client, &clientSize);
    if (clientfd == -1) {
        cout << "Failed to accept connection." << endl;
        return 1;
    }

    while(1){
        cout<<"hi"<<endl;
        // if ((bytes = recv(sokt, iptr, imgSize , MSG_WAITALL)) == -1) {
        //     std::cerr << "recv failed, received bytes = " << bytes << std::endl;
        // }
        uint32_t size = 0;
        if (recv(clientfd, &size, sizeof(size), MSG_WAITALL) != sizeof(size)) {
            cout << "Failed to receive size." << endl;
            break;
        }
        
        vector<char> buffer(size);
        if (recv(clientfd, buffer.data(), size, MSG_WAITALL) != size) {
            cout << "Failed to receive data." << endl;
            break;
        }
        
        Mat image = imdecode(buffer, IMREAD_COLOR);
        if (image.empty()) {
            cout << "Failed to decode image." << endl;
            break;
        }


        imshow("test", image);

        waitKey(1);
    }

    return 0;
}