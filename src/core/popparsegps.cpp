#include "core/popparsegps.hpp"
#include <iostream>


//#include <boost/lexical_cast.hpp>

using namespace std;

//#define POPPARSEGPS_VERBOSE

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

PopParseGPS::PopParseGPS(unsigned notused) : PopSink<char>("PopParseGPS", 1), headValid(false), gpsFix(false), lat(0.0), lng(0.0)
{

}

void PopParseGPS::process(const char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{
	char c = data[0];

	if( !headValid )
	{
		if( c == '\n' )
			headValid = true;
	}
	else
	{

		if( c == '\n' )
		{
			parse();
			command.erase(command.begin(),command.end());
		}
		else
		{
			command.push_back(c);
		}
	}
}


unsigned parseHex(std::string &in)
{
	unsigned result;
	std::stringstream ss;
	ss << std::hex << in;
	ss >> result;
	return result;
}

unsigned parseInt(char *in)
{
	unsigned result;
	std::stringstream ss;
	ss << in;
	ss >> result;
	return result;
}

double parseDouble(std::string &in)
{
	double result;
	std::stringstream ss;
	ss << in;
	ss >> result;
	return result;
}

bool checksumOk(std::string &str, unsigned len)
{
	char givenChecksum;

	if( len < 5 )
		return false;

	std::string checkStr = str.substr(len - 3, 2);

	givenChecksum = parseHex(checkStr);


	// this extracts bytes according to http://www.gpsinformation.org/dale/nmea.htm
	std::string checkedBytes = str.substr(1, len - 5);

	char calculatedChecksum = 0x00;

	// loop and xor
	for(char& c : checkedBytes) {
		calculatedChecksum ^= c;
	}

	//    	cout << "bytes: " << checkedBytes << endl;
	//    	cout << "checksum: " << (int) checksum << endl;
	//    	cout << "checksum bytes: " << checkStr << endl;
	//    	cout << "given: " << (int) givenChecksumInt << " calculated " << (int) calculatedChecksum << endl;


	return (givenChecksum == calculatedChecksum);
}

double parseRetardedFormat(std::string &str, bool positive)
{
	size_t found;

	found = str.find(".");

	if( found == std::string::npos )
	{
		cout << "GPS format missing decimal" << endl;
		return 0.0;
	}

	if( found < 2 )
	{
		cout << "LHS of GPS format is too small (" << found << ")" << endl;
		return 0.0;
	}

	std::string minute = str.substr(found-2, str.length() - 1);
	std::string degree = str.substr(0, found-2);

	double sign = (positive)?1:-1;

//	cout << "Minute is: " << minute << " degree is: " << degree << endl;

	return (parseDouble(degree) + (parseDouble(minute)/60)) * sign;
}

bool fixStatusOk(int status) {
	if( status < 1 )
		return false;

	if( status > 6 )
		return false;

	return true;
}

void PopParseGPS::setFix(double lat, double lng, double time)
{
	boost::lock_guard<boost::mutex> guard(mtx_);
	this->lat = lat;
	this->lng = lng;
	//time
	this->gpsFix = true;
}



void PopParseGPS::gga(std::string &str)
{
	char seps[] = ",";
	char *token;
	char *state;
	unsigned index = 0;
	int fixStatus = -1;
	std::string latStr, lngStr;
	bool latPositive = true, lngPositive = true;

	token = strtok_r_single( &str[0], seps, &state );
	while( token != NULL )
	{
#ifdef POPPARSEGPS_VERBOSE
		cout << index << ": " << token << endl;
#endif

		switch(index)
		{
			case 2:
				latStr = std::string(token);
				break;
			case 3:
				latPositive = (strncmp("N", token, 1)==0)?true:false;
				break;
			case 4:
				lngStr = std::string(token);
				break;
			case 5:
				lngPositive = (strncmp("E", token, 1)==0)?true:false;
				break;
			case 6:
				fixStatus = parseInt(token);
#ifdef POPPARSEGPS_VERBOSE
				cout << "Fix status: " << fixStatus << endl;
#endif
				break;
			default:
				break;
		}


		token = strtok_r_single( NULL, seps, &state );
		index++;
	}

	if( fixStatusOk(fixStatus) )
	{
		setFix(parseRetardedFormat(latStr, latPositive), parseRetardedFormat(lngStr, lngPositive), 0);
	}

//	cout << boost::lexical_cast<string>( parseRetardedFormat(latStr, latPositive) ) << endl;
//	cout << boost::lexical_cast<string>( parseRetardedFormat(lngStr, lngPositive) )<< endl;
//	cout << "Fix ok: " << fixOk(fixStatus) << endl;
//	cout << latStr << latPositive << lngStr << lngPositive << endl;
}

bool PopParseGPS::gpsFixed()
{
	return gpsFix;
}
boost::tuple<double, double, double> PopParseGPS::getFix()
{
	boost::lock_guard<boost::mutex> guard(mtx_);
	return boost::tuple<double, double, double>(this->lat, this->lng, 0.0);
}

void PopParseGPS::parse()
{
	std::size_t found;
	bool cok = false;
	unsigned len = command.size();
	if( len == 0 )
		return;

	// str will contain the entire message, with a trailing \r (from \r\n)
	std::string str(command.begin(),command.end());

	cok = checksumOk(str, len);

	if( !cok ) {
//		cout << "bad GPS checksum (" << str << ")" << endl;
		return;
	}


#ifdef POPPARSEGPS_VERBOSE
	cout << "command: " << str << endl;
#endif

	found = str.find("$GPGSA");
	if( found == 0 )
	{

	}

	found = str.find("$GPRMC");
	if( found == 0 )
	{

	}

	found = str.find("$GPGGA");
	if( found == 0 )
	{
		gga(str);
	}

}


} //namespace

