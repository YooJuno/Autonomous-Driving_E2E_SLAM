  // g++ -o drive drive.cpp $(pkg-config --libs --cflags opencv4)
#define PAUSE 0
#define RECORD 1

#define FPS 20

#include <iomanip>
#include <iostream>
#include <termios.h>
#include <cstdlib>
#include <fstream> 
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <unistd.h>   /* For open(), creat() */
#include "opencv4/opencv2/opencv.hpp"

using namespace cv;
using namespace std;
using namespace dnn;

void openSerialPort();
void save_driving_log(std::string, std::string);
//int yolo(cv::Mat img_name);

int fd;
int speed = 0;
char key;

char go[1]= {'w'};
char left[1] = {'a'};
char right[1] = {'d'};
char stop[1] = {'s'};
char back[1] = {'x'};
char center[1] = {'f'};

int main(int argc, char** argv){

    //open serial port
    openSerialPort();

    if(argc <3){
        std::cout << "Usage: ./test PATH_TO_SAVE [0|1](0:drive_mode, 1:logging_mode)"<<std::endl;
        exit(-1);
    }

    int recording = atoi(argv[2]);
    std::string save_path = argv[1];

    std::string output_frame_file_path = save_path + "/frame";
    std::string output_log_file_path = save_path + "/driving_log.csv";
    std::string driving_log;
    int frame_no = 0;

    cv::VideoCapture cap(2);

    if(!cap.isOpened()){
        std::cout << "can not open the camera..." << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    std::cout<<cap.get(cv::CAP_PROP_FRAME_WIDTH)<<endl;
    std::cout<<cap.get(cv::CAP_PROP_FRAME_HEIGHT)<<endl;


    float angle = 0.0;
    //int speed_count = 0;

    char go[1]= {'w'};
    char left[1] = {'a'};
    char right[1] = {'d'};
    char stop[1] = {'s'};
    char back[1] = {'x'};
    char center[1] = {'f'};
    
    cv::Mat img;
    cv::Mat img_pre;

    int check = 2; // 기본 속도
    int count = 0; // frame maintain

    while(1){
        cap >> img_pre;

        // cout << "\n[speed " << speed << "] [check " << check << "]" << endl;
        cv::imshow("camera img", img_pre); //show the frame

        //key 값 받기
        key = cv::waitKey(1000/FPS);

        if(key == 'w'){
            write(fd, go, 1);
            speed++;
            // write(fd, go, 1);
            // speed++;
            // write(fd, go, 1);
            // speed++;
        } 
        else if(key == 'd'){
            angle += 0.25;
            if(angle >= 1)
                angle = 1;
            write(fd, right, 1);
        } 
        else if(key == 'a'){
            angle -= 0.25;
                if(angle <= -1)
                    angle = -1;
            write(fd, left, 1);
        } 
        else if(key == 's'){
            angle = 0;
            speed = 0;
            write(fd, stop, 1);
        } 
        else if(key == 'x'){
            write(fd, back, 1);
        } 
        else if(key == 'f'){
            angle = 0;
            write(fd, center, 1);
        } 
        else if(key == 27){
            write(fd, stop, 1);
            break;
        }
        else if(key == 'p'){
            recording = PAUSE;
        } 
        else if(key == 'r'){
            recording = RECORD;
        }
        else{
           key = 0;
        }
        
        std::string file_name = output_frame_file_path + std::to_string(frame_no) + ".jpg";
        driving_log = file_name + ',' + std::to_string(angle) + ',' + std::to_string(speed);
        std::cout << file_name << ", " << angle << ", " << speed << std::endl;
        //std::cout << "key: " << key << std::endl;
        
        if(recording == RECORD){
            
            //cv::imwrite(file_name, img);
            cv::imwrite(file_name, img_pre);
            save_driving_log(output_log_file_path, driving_log);
            frame_no ++;
        }

        //cout << "count " << count << endl;
    }
    std::cout << "done.." << std::endl;
    write(fd, stop, 1);
    return 0;
}

void openSerialPort()
{
    struct termios toptions;

    /* open serial port */
    fd = open("/dev/ttyACM0", O_RDWR | O_NOCTTY);   
    std::cout << "FD: " << fd << std::endl;

    /* get current serial port settings */
    tcgetattr(fd, &toptions);

    /* set 9600 baud both ways */
    cfsetispeed(&toptions, B9600);
    cfsetospeed(&toptions, B9600);

    /* 8 bits, no parity, no stop bits */
    toptions.c_cflag &= ~PARENB;
    toptions.c_cflag &= ~CSTOPB;
    toptions.c_cflag &= ~CSIZE;
    toptions.c_cflag |= CS8;

    /* Canonical mode */
    toptions.c_lflag |= ICANON;

    /* commit the serial port settings */
    tcsetattr(fd, TCSANOW, &toptions);
}

void save_driving_log(std::string file_name, std::string driving_log){
    std::ofstream fout;
    fout.open(file_name, std::ios_base::out | std::ios_base::app);
    fout << driving_log << std::endl;
    fout.close();
}

/*
시작할 때 w 누르고 시작 (자동으로 속도 2 고정) - 바로 r 눌러서 기록 시작
사람이 멀리 있을 때(보행자 표지판), 속도 1로 바뀜. 이후 속도 2, 0 자동으로 바뀜.
사람이 가까이 있을 때(stop 표지판), 속도 0으로 바뀜. stop 표지판 없어지면 바로 w 누르기. (속도 2로 바뀜)
끝낼 때 p 누르고 esc.
*/

/*
w 앞으로 속도 조절
s 멈춤
a d 똑같음
f 중앙으로 방향 정렬(11자로, 속도 안바뀜)

실행할 때 컴파일 ./test 이미지저장할경로 (0 or 1)
0 : 이미지 저장을 안함
1 : 함(주행하다가 모드 바꿀 수 있음) 기록을 하다가 P 누르면 기록이 정지가 됨 정지가 되도 주행은 가능 . 코너 집중할 때 갔다가 다시 빠꾸쳐서 다시 ㄱ. R 누르면 기록 다시 시
*/