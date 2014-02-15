#ifndef __POP_PARSE_GPS_HPP_
#define __POP_PARSE_GPS_HPP_


#include <core/popsink.hpp>
#include <core/popsource.hpp>
#include <iostream>
#include <string>

using namespace std;


namespace pop
{

char* strtok_r_single(
    char *str,
    const char *delim,
    char **nextp)
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




class PopParseGPS : public PopSink<unsigned char>
{
public:
//	 PopSink<unsigned char> *tx; // PopSink must be inherited b/c of virtual classes, so we fake out a pointer here
//	 PopSource<unsigned char> rx;
	 bool headValid;
	 std::vector<unsigned char> command;
	 bool gpsFix;
	 double lat;
	 double lng;

private:
	// pointer to a single timestamp given to us in the previous call to process
//	const PopTimestamp* previous_timestamp;
//
//	// the number of data samples from the previous call to process
//	size_t previous_size;
public:
	 PopParseGPS(unsigned notused) : PopSink<unsigned char>("PopParseGPS", 1), headValid(false), gpsFix(false), lat(0.0), lng(0.0)
    {
//		 tx = this;
    }
    void init() {}
    void process(const unsigned char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
    {
//    	if( size != 1 ) {
//    		cout << "Error " << this->get_name() << " may only accept 1 character at a time";
//    		return;
//    	}
//
//    	cout << "here" << endl;

    	char c = data[0];
//    	cout << c;
//
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

    void parse()
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

    void gga(std::string &str)
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

    void execute(std::string &method, json_token *tokens, int methodId)
    {
//     	const struct json_token *tok;
//    	if( method.compare("log") == 0 )
//    	{
//    		tok = find_json_token(tokens, "params[0]");
//    		if( tok && tok->type == JSON_TYPE_STRING )
//    		{
//    			rcp_log(std::string(tok->ptr, tok->len));
//    			respond_int(0, methodId);
//    		}
//    	}
//
//    	if( method.compare("count") == 0 )
//    	{
//    		int ret = rpc_count();
//    		respond_int(ret, methodId);
//    	}
    }

};

}


#endif
