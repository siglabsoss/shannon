#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/lexical_cast.hpp>


#include "core/popparsegps.hpp"
#include "core/geohelper.hpp"
#include "core/utilities.hpp"


using namespace std;
using namespace boost::gregorian;

//#define POPPARSEGPS_VERBOSE

namespace pop
{

PopParseGPS::PopParseGPS(unsigned notused) : PopSink<char>("PopParseGPS", 1), headValid(false), gpsFix(false), lat(0.0), lng(0.0), tx("PopParseGPStx")
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

std::string PopParseGPS::get_checksum(std::string str)
{
	unsigned len = str.length();
	if( len < 5 )
	{
		cout << "checksum string " << str << " is too short" << endl;
		return "";
	}

	std::string checkedBytes = str.substr(1, len-1);

//	cout << "checking: " << checkedBytes;

	char calculatedChecksum = 0x00;

	// loop and xor
	for(char& c : checkedBytes) {
		calculatedChecksum ^= c;
	}

	char buf[16];

	// convert to hex this way because hex is so annoying in cpp
	snprintf(buf, 16, "%02x", (int)calculatedChecksum);

	return std::string(buf);
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

//	cout << str << endl;

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















// https://github.com/martinmm/tools/blob/23520f248ed36f08fbe86e4eff01f27307bcfd23/itow_conv/itow_conv.c


#define SECS_DAY    (60*60*24)
#define SECS_WEEK   (60*60*24*7)
#define SECS_YEAR   (SECS_DAY*365)

#define isleap(x) ((((x)%400)==0) || (!(((x)%100)==0) && (((x)%4)==0)))

const int8_t DAYS_MONTH[12] =
{
    31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
};


// unused for now
void itow2time (uint16_t  gps_week,
                uint32_t  gps_itow,
                uint8_t*  gps_second,
                uint8_t*  gps_minute,
                uint8_t*  gps_hour,
                uint8_t*  gps_day,
                uint8_t*  gps_month,
                uint16_t* gps_year)
{
    uint32_t total_seconds, total_days, month_days, years=0, months=0;
    uint32_t gps_tow = gps_itow / 1000;

    /* sanity check */
    if (gps_tow >= SECS_WEEK) return;

    /* seconds since gps start 6-JAN-1980 00:00:00 */
    total_seconds = (uint32_t)(gps_tow + gps_week*SECS_WEEK);

    /* days since 1-JAN-1980 */
    total_days = (total_seconds / (SECS_DAY)) + 5;

    /* years since 1980 */
    while (1)
    {
        if (total_days < (365 + isleap(1980 + years))) break;
        total_days -= (365 + isleap(1980 + years));
        years++;
    }

    /* months since start of year */
    while (1)
    {
        if ((isleap(years)) && (months == 1))
            month_days = 29;
        else
            month_days = DAYS_MONTH[months];

        if (total_days < month_days) break;

        total_days -= month_days;
        months++;
    }

    /* convert */
    *gps_hour   = (uint8_t) ((total_seconds % SECS_DAY) / 3600);
    *gps_minute = (uint8_t) ((total_seconds % 3600) / 60);
    *gps_second = (uint8_t) ((total_seconds % 60));
    *gps_month  = (uint8_t) (months + 1);
    *gps_day    = (uint8_t) (total_days + 1);
    *gps_year   = (uint16_t)(1980 + years);
}




// http://www.novatel.com/support/knowledge-and-learning/published-papers-and-documents/unit-conversions/
// http://adn.agi.com/GNSSWeb/
// http://www.boost.org/doc/libs/1_39_0/doc/html/date_time/examples.html
boost::tuple<uint32_t, uint32_t> PopParseGPS::gps_now()
{
	// these two lines actually read the system clock twice
	PopTimestamp now = get_microsec_system_time();
	date date_now(day_clock::universal_day());

	uint64_t epoc_seconds = now.get_full_secs();
	uint64_t midnight_seconds = epoc_seconds % SECS_DAY;

	date gps_start = from_simple_string("1980-01-06"); // start date of GPS clocks
	days delta = date_now-gps_start;

	uint64_t seconds_total = delta.days() * SECS_DAY;  // number of seconds in the full days since jan 6th, 1980

	double weeks_total = seconds_total/(double)SECS_WEEK;

	double weeks_total_fraction = weeks_total - floor(weeks_total);

	uint32_t gps_weeks = floor(weeks_total);

	uint32_t days_into_week = round(weeks_total_fraction * 7); // 7 days a week

	uint32_t secs_into_week = days_into_week * SECS_DAY;

	uint32_t gps_secs = secs_into_week + midnight_seconds;

//			cout << "secs_total: " << seconds_total << endl;
//			cout << "weeks_total: " << weeks_total << endl;
//			cout << "gps_weeks: " << gps_weeks << endl;  // NEEDED
//			cout << "weeks_total_fraction: " << weeks_total_fraction << endl;
//			cout << "days_into_week: " << days_into_week << endl;
//			cout << "secs_into_week: " << secs_into_week << endl;
//			cout << "midnight_seconds: " << midnight_seconds << endl;
//			cout << "gps_secs: " << gps_secs << endl; // NEEDED

	return boost::make_tuple(gps_weeks, gps_secs);
}


void PopParseGPS::set_debug_on()
{
	std::string msg("\r\n$PSRF105,1*3E\r\n");
	this->tx.process(msg.c_str(), msg.length());
}

void PopParseGPS::set_debug_off()
{
	std::string msg("\r\n$PSRF105,0*3F\r\n");
	this->tx.process(msg.c_str(), msg.length());
}


void PopParseGPS::hot_start()
{
	GeoHelper geo_helper_;

	double lat,lng, alt;

	lat = 37.477083;
	lng = -122.196742;
	alt = 0.0;

	uint32_t gps_weeks, gps_secs;
	boost::tie(gps_weeks, gps_secs) = gps_now();

	uint32_t clk_offset = 0; // 0 is use saved value
	uint32_t channel_count = 12; // number of channels to use, 1-12
	uint16_t reset_cfg = 0x01;  // sigle byte wide bit mask.  Using a uint8_t can confuse ostringstream into printing a character

	ostringstream os;
	os << "$PSRF104," << boost::lexical_cast<string>(lat) << ',' << boost::lexical_cast<string>(lng) << ',' << boost::lexical_cast<string>(alt) << ',' << clk_offset << ',' << gps_secs << ',' << gps_weeks << ',' << channel_count << ',' << reset_cfg;

//	cout << os.str() << endl;

	// call get_checsum without the trailing * or anything
	std::string checksum = get_checksum(os.str());

	// now add the * and checksum to the ostringstream
	os << '*' << checksum;

//	cout << "Final Message:" << endl;
//	cout << os.str();

	ostringstream msg;

	// add on new lines
	msg << "\r\n" << os.str() << "\r\n";

	this->tx.process(msg.str().c_str(), msg.str().length());
}


} //namespace

