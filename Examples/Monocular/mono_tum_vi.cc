
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

// TCP header
#include "opencv4/opencv2/opencv.hpp"
// #include <iostream>
#include <sys/socket.h> 
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h> 
#include <string.h>
#include <pthread.h>



#include<System.h>

#define _WEBCAM_BUILD_

using namespace std;
using namespace cv;

// #define PORT 8485

int main(int argc, char **argv)
{
#ifdef _WEBCAM_BUILD_

    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        cout << "Failed to create socket." << endl;
        return 1;
    }
    
    sockaddr_in server;
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port = htons(atoi(argv[3]));
    
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
    
    





    
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
    cout << endl << "-------" << endl;
    

#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point initT = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point initT = std::chrono::monotonic_clock::now();
#endif


    // Main loop
    while(true)
    {


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
        
        Mat frame = imdecode(buffer, IMREAD_COLOR);
        if (frame.empty()) {
            cout << "Failed to decode image." << endl;
            break;
        }

        







#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point nowT = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point nowT = std::chrono::monotonic_clock::now();
#endif
        // Pass the image to the SLAM system
        Sophus::SE3f juno_tcw = SLAM.TrackMonocular(frame, std::chrono::duration_cast<std::chrono::duration<double> >(nowT-initT).count());
        // Eigen::Matrix4f T = juno_tcw.matrix(); // Convert Sophus::SE3f to Eigen::Matrix4f
        // double x = T(0, 3); // Extract x value from translation part
        // double z = T(2, 3); // Extract z value from translation part

        Eigen::Matrix<float,3,1> mOw = juno_tcw.translation();
        double x = mOw[0]; // Extract x value from translation part
        double z = mOw[2]; // Extract z value from translation part

        string msg = to_string(x) + "," + to_string(z);


        // 이미지 수신 후 클라이언트에게 "check" 메시지 보내기
        if (send(clientfd, msg.c_str(), msg.length(), 0) != msg.length()) {
            cout << "Failed to send data." << endl;
            break;
        }
    }
        
    // Stop all threads
    SLAM.Shutdown();

    //slam->SaveSeperateKeyFrameTrajectoryTUM("KeyFrameTrajectory-1.txt", "KeyFrameTrajectory-2.txt", "KeyFrameTrajectory-3.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

#else
    
#endif
    return 0;
}