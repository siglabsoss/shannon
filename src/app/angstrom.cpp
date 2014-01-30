#include <stdio.h>      // standard input / output functions
#include <stdlib.h>
#include <string.h>     // string function definitions
#include <unistd.h>     // UNIX standard function definitions
#include <fcntl.h>      // File control definitions
#include <errno.h>      // Error number definitions
#include <termios.h>    // POSIX terminal control definitions
#include <iostream>


using namespace std;

void setupUart(int USB)
{
struct termios tty;
struct termios tty_old;
memset (&tty, 0, sizeof tty);

/* Error Handling */
if ( tcgetattr ( USB, &tty ) != 0 )
{
cout << "Error " << errno << " from tcgetattr: " << strerror(errno) << 
endl;
}

/* Save old tty parameters */
tty_old = tty;

/* Set Baud Rate */
cfsetospeed (&tty, (speed_t)B115200);
cfsetispeed (&tty, (speed_t)B115200);

/* Setting other Port Stuff */
tty.c_cflag     &=  ~PARENB;        // Make 8n1
tty.c_cflag     &=  ~CSTOPB;
tty.c_cflag     &=  ~CSIZE;
tty.c_cflag     |=  CS8;

tty.c_cflag     &=  ~CRTSCTS;       // no flow control
tty.c_cc[VMIN]      =   1;                  // read doesn't block
tty.c_cc[VTIME]     =   5;                  // 0.5 seconds read timeout
tty.c_cflag     |=  CREAD | CLOCAL;     // turn on READ & ignore ctrl lines

/* Make raw */
cfmakeraw(&tty);

/* Flush Port, then applies attributes */
tcflush( USB, TCIFLUSH );
if ( tcsetattr ( USB, TCSANOW, &tty ) != 0)
{
cout << "Error " << errno << " from tcsetattr" << endl;
}
}

void writeUart(int handle, char bytes[], unsigned length) {
  int n_written = 0;

write( handle, bytes, length);
return;

  do {
    n_written += write( handle, &bytes[n_written], 1 );
  }
  while (bytes[n_written-1] != '\r' && n_written > 0);

}


int main(void) {

int USB = open( "/dev/ttyO4", O_RDWR| O_NOCTTY );

setupUart(USB);

char cmd[] = "Hello World\r\n";

writeUart(USB, cmd, strlen(cmd));




}
