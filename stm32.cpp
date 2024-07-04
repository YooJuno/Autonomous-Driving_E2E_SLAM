#include "mbed.h"

BufferedSerial board(PA_9, PA_10, 9600);
BufferedSerial pc(USBTX, USBRX, 9600);

char str[10];
char message[80];
char velocity[5];
char angle[5];

char _pre[1] = {0x02};
char _mid[1] = {0x03};
char _last[1] = {0x00};

int main()
{
    sprintf(message, "pc connected! \r\n");
    pc.write(message, strlen(message));
    
    // db
    sprintf(velocity, "3B6"); // 정지 속도
    
    //char angleData[5][5] = {"223",  "2AF" , "361", "40D", "4DF"};
    //char angleData[5][5] = {"223",  "2D1" , "381", "42F", "4DF"}; // Black Car
    
    //char angleData[5][5] = {"223",  "2B2" , "341", "410", "4DF"};
    char angleData[9][5] = {"223","27A", "2D1", "328", "381", "3D7", "42F", "487", "4DF"}; // 547 634 721 808 897! 983 1071 1159 1247
    char velocityData[6][5] = {"390","396" ,"3B6", "3E1", "40C", "439"}; // 912 918 950 985 998 1024 ->  950   993(3E1)  1036(40C)  1081(439)
    
    sprintf(angle, "381"); // 차의 기본 각도
    
    char mode = 'P';
    //int num = 2;
    int num = 4;
    int vel_num = 2;
    
    bool automoving = false; // true면 자율주행으로 온 값, false면 키보드 입력으로 온 값
    
    sprintf(velocity, "3B6");
    sprintf(angle, "381");
    
    sprintf(message, "initialize vel and angle ...\r\n");
    pc.write(message, strlen(message));
    sprintf(message, "init velocity: %s\r\ninit angle: %s\r\n", velocity, angle);
    pc.write(message, strlen(message));
    
    ThisThread::sleep_for(500ms);
    board.write(_pre, 1);
    board.write(velocity, strlen(velocity));
    board.write(angle, strlen(angle));
    board.write("01", 2);        
    board.write(_mid, 1);
    board.write(_last, 1);   
    
    while(1)
    {
        
        if(pc.readable())
        {
            pc.read(str, sizeof(str));
            mode = str[0];
            sprintf(message,"get %c\r\n", mode);
            pc.write(message, strlen(message));

            switch(mode)
            {
                case 'w' :
                {
                    memset(str, 0, sizeof(str));
                    vel_num++;
                    if (vel_num > 5) vel_num = 5;
                    memcpy(velocity, velocityData[vel_num], sizeof(velocityData[vel_num]));
                    mode = 'P';
                    break;
                }
                case 'a' :
                {
                    memset(str, 0, sizeof(str));
                    num--;
                    if (num < 0) num = 0;
                    memcpy(angle, angleData[num], sizeof(angleData[num]));
                    mode = 'P';
                    break;
                }
                case 'd' :
                {
                    memset(str, 0, sizeof(str));
                    num++;
                    if (num > 8) num = 8;
                    memcpy(angle, angleData[num], sizeof(angleData[num]));
                    mode = 'P';
                    break;
                }
                case 'x' :
                {
                    memset(str, 0, sizeof(str));
                    vel_num--;
                    if (vel_num < 0) vel_num = 0;
                    memcpy(velocity, velocityData[vel_num], sizeof(velocityData[vel_num]));
                    mode = 'P';
                    break;
                }
                case 's' :
                {
                    memset(str, 0, sizeof(str));
                    sprintf(velocity, "3B6");
                    sprintf(angle, "381");
                    num = 4;
                    vel_num = 2;                
                    break;
                }
                case 'f' :
                    memset(str, 0, sizeof(str));
                    sprintf(angle, "381");
                    num = 4;
                     
                default :
                {
                    memset(str, 0, sizeof(str));
                    break;
                }
            }        
        }
  
        //board.write(_pre , 1);
        
        // sprintf(message, "send velocity: ");
        // pc.write(message, strlen(message));
        //board.write(velocity, strlen(velocity));
        //pc.write(velocity, strlen(velocity));
        
        //sprintf(message, "\r\nsend angle: ");
        //pc.write(message, strlen(message));
        //board.write(angle, strlen(angle));
        //pc.write(angle, strlen(angle));
        
        //sprintf(message, "\r\n");
        //pc.write(message, strlen(message));
        board.write(_pre , 1);
        board.write(velocity, strlen(velocity));
        board.write(angle, strlen(angle));
        board.write("01", 2);        
        board.write(_mid, 1);
        board.write(_last, 1);   
        
        sprintf(message, "\r\nVELOCITY: "); 
        pc.write(message, strlen(message));
        pc.write(velocity, strlen(velocity));
        
        sprintf(message, "\r\nANGLE: ");
        pc.write(message, strlen(message));
        pc.write(angle, strlen(angle));
        
        sprintf(message, "\r\n\r\n------------------\r\n\r\n"); 
        pc.write(message, strlen(message));
                
        //ThisThread::sleep_for(500ms);  //for debugging, 
        ThisThread::sleep_for(0ms);    
        
        
    }

}


