#include "core/popparsegps.hpp"
#include <iostream>


using namespace std;



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

PopParseGPS::PopParseGPS(unsigned notused) : PopSink<unsigned char>("PopParseGPS", 1), headValid(false), gpsFix(false), lat(0.0), lng(0.0)
{

}

void PopParseGPS::process(const unsigned char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
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

bool checksumOk(std::string &str, unsigned len)
{
	unsigned char givenChecksum;
	std::string checkStr = str.substr(len - 3, 2);

	givenChecksum = parseHex(checkStr);


	// this extracts bytes according to http://www.gpsinformation.org/dale/nmea.htm
	std::string checkedBytes = str.substr(1, len - 5);

	unsigned char calculatedChecksum = 0x00;

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

void PopParseGPS::gga(std::string &str)
{
	char seps[] = ",";
	char *token;
	char *state;
	unsigned index = 0;
	int fixStatus = -1;

	token = strtok_r_single( &str[0], seps, &state );
	while( token != NULL )
	{
		cout << index << ": " << token << endl;

		if( index == 6 )
		{
			fixStatus = parseInt(token);
			cout << "Fix status: " << fixStatus << endl;
		}


		/* Do your thing */
		token = strtok_r_single( NULL, seps, &state );
		index++;
	}
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
		cout << "bad GPS checksum" << endl;
		return;
	}



	cout << "command: " << str << endl;

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

