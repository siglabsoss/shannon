#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/asio.hpp>


#include "core/utilities.hpp"
#include "b64/b64.h"


namespace pop
{

char* strtok_r_single(char *str, const char *delim, char **nextp)
{
	char *ret;

	if (str == NULL)
	{
		str = *nextp;
	}

	// removing this line changes this from strtok_r to strtok_r_single
	//    str += strspn(str, delim);

	if (*str == '\0')
	{
		return NULL;
	}

	ret = str;

	str += strcspn(str, delim);

	if (*str)
	{
		*str++ = '\0';
	}

	*nextp = str;

	return ret;
}


bool operator==(const uuid_t& lhs, const uuid_t& rhs)
{
    return (lhs.parts.UIDMH == rhs.parts.UIDMH) && (lhs.parts.UIDML == rhs.parts.UIDML) && (lhs.parts.UIDL == rhs.parts.UIDL);
}


uuid_t b64_to_uuid(std::string b64_serial)
{
	uuid_t result;

	unsigned encodedCount = b64_serial.length();
	char serialDecoded[encodedCount];
	unsigned decodedCount;
	b64_decode(b64_serial.c_str(), encodedCount, serialDecoded, &decodedCount);

	if( decodedCount != sizeof(uuid_t) )
	{
		std::cout << "Invalid serial length" << std::endl;
		result.parts.UIDMH = result.parts.UIDML = result.parts.UIDL = 0;
		return result;
	}

	// #lazy
	b64_decode(b64_serial.c_str(), encodedCount, (char*)&result, &decodedCount);

	return result;
}

std::string uuid_to_b64(uuid_t u)
{

	unsigned encodedCount;
	// b64_length_encoded() tells us the worst case size for the b64 string, we need 1 more char
	char b64_encoded[b64_length_encoded(sizeof(uuid_t))+1];

	// b64 encode data
	b64_encode((char*)&u, sizeof(uuid_t), b64_encoded, &encodedCount);

	// pack in a null so we can %s with printf
	b64_encoded[encodedCount] = '\0';

	return std::string(b64_encoded);
}


// code pulled from '/home/joel/uhd/host/lib/types/time_spec.cpp
// because that file was compiled with incorrect flags and get_system_time() returns garbage
namespace pt = boost::posix_time;
PopTimestamp get_microsec_system_time(void){
	pt::ptime time_now = pt::microsec_clock::universal_time();
	pt::time_duration time_dur = time_now - pt::from_time_t(0);
	return PopTimestamp(
			time_t(time_dur.total_seconds()),
			long(time_dur.fractional_seconds()),
			double(pt::time_duration::ticks_per_second())
	);
}


int getch(void)
{
  int ch;
  struct termios oldt;
  struct termios newt;
  tcgetattr(STDIN_FILENO, &oldt); /*store old settings */
  newt = oldt; /* copy old settings to new settings */
  newt.c_lflag &= ~(ICANON | ECHO); /* make one change to old settings in new settings */
  tcsetattr(STDIN_FILENO, TCSANOW, &newt); /*apply the new settings immediatly */
  ch = getchar(); /* standard getchar call */
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt); /*reapply the old settings */
  return ch; /*return received char */
}

// http://stackoverflow.com/questions/448944/c-non-blocking-keyboard-input
int kbhit(void)
{
    struct timeval tv = { 0L, 0L };
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(0, &fds);
    return select(1, &fds, NULL, NULL, &tv);
}

std::string pop_get_hostname(void)
{
	static bool set = false;
	char buf[256];

	if(!set)
	{
		int ret = gethostname(buf, 256);
		if( ret != 0 )
		{
			std::cout << "couldn't read linux hostname!" << std::endl;
			strncpy(buf, "unkown", 256);
		}
		set = true;
	}

	static std::string hostname(buf);

	return hostname;
}


}
