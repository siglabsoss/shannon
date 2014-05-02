#include <iostream>
#include "core/poppackethandler.hpp"
#include "core/util.h"

//extern "C" {
#include "dsp/prota/popsparsecorrelate.h"
//}

using namespace std;


namespace pop
{


PopPacketHandler::PopPacketHandler(unsigned notused) : PopSink<uint32_t>("PopPacketHandler", 1)
{

}

void PopPacketHandler::process(const uint32_t* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{
//	cout << "got " << size << " things" << endl;



	uint32_t comb[] = {0, 343200, 559680, 601920, 755040, 813120, 929280, 955680, 997920, 1003200, 1029600, 1135200, 1193280, 1240800, 1251360, 1383360, 1404480, 1483680, 1520640, 1647360, 1694880, 1800480, 1879680, 1921920, 1932480, 1958880, 2085600, 2122560, 2164800, 2180640, 2196480, 2244000, 2344320, 2428800, 2434080, 2476320, 2550240, 2872320, 3067680, 3278880, 3410880, 3669600, 3738240, 3806880, 3838560, 3944160, 3986400, 4134240, 4239840, 4297920, 4345440, 4414080, 4419360, 4593600, 4678080, 4736160, 4878720, 4894560, 5116320, 5221920, 5253600, 5290560, 5512320, 5639040, 5834400, 6019200, 6225120, 6383520, 6452160, 6494400, 6600000, 6668640, 6916800, 7138560, 7170240, 7186080, 7223040, 7275840, 7370880, 7571520, 7587360, 7597920, 7751040, 7898880, 7904160, 7930560, 8110080, 8310720, 8469120, 8500800, 8580000, 8748960, 8880960, 8954880, 8986560, 9086880, 9150240, 9176640, 9229440, 9451200, 9572640, 9625440, 9757440, 9884160, 10047840, 10142880, 10243200};
	uint32_t left[] = {4121674, 4135690, 4142026, 4144714, 4487818, 4704586, 4746826, 4899274, 4957642, 5073034, 5100106, 5141962, 5148106, 5174026, 5280202, 5337034, 5385802, 5395594, 5527690, 5548810, 5628490, 5664394, 5792266, 5839498, 5945290, 6023818, 6066634, 6077194, 6103306, 6229642, 6267466, 6309514, 6325834, 6341194, 6389002, 6488650, 6573322, 6578698, 6621130, 6694474, 7017034, 7212106, 7423114, 7556170, 7814218, 7882378, 7951498, 7983370, 8089162, 8130634, 8278858, 8384650, 8442442, 8489098, 8559178, 8563786, 8737930, 8822602, 8880778, 9023050, 9039946, 9261322, 9366922, 9398026, 9435082, 9656458, 9783562, 9979018, 10164298, 10369930, 10528522, 10596490, 10638922, 10743946, 10814026, 11061898, 11284042, 11314762, 11330506, 11367370, 11420746, 11516362, 11716426, 11731594, 11743306, 11895562, 12043210, 12048778, 12075274, 12254794, 12456010, 12613642, 12646474, 12725002, 12893770, 13025866, 13099786, 13130890, 13231882, 13295050, 13321162, 13373578, 13596106, 13717834, 13770826, 13901770, 14029258, 14192074, 14288458, 14405578, 14410186, 14411914, 14419018, 14420746, 14436106, 14438410, 14445514, 14446858, 14450122, 14451850, 14458954, 14461066, 14468170, 14470282};
//	uint32_t right[] = {4137127, 4141927, 4143847, 4487335, 4704295, 4745767, 4899367, 4957351, 5074279, 5100199, 5142247, 5148199, 5174119, 5278759, 5338855, 5384743, 5395495, 5527399, 5549287, 5627623, 5666023, 5791591, 5839591, 5944999, 6024871, 6065383, 6077287, 6103207, 6231271, 6266599, 6309223, 6324583, 6341287, 6387943, 6489319, 6573415, 6578407, 6619879, 6695143, 7015975, 7213159, 7423015, 7555303, 7814311, 7882471, 7951399, 7983271, 8088295, 8131303, 8278567, 8384551, 8442343, 8490535, 8557735, 8564647, 8737831, 8823655, 8880295, 9023719, 9038887, 9261607, 9366247, 9398311, 9434983, 9657319, 9783655, 9979111, 10163431, 10370599, 10527271, 10596583, 10639015, 10745191, 10813159, 11061607, 11282791, 11315431, 11330599, 11367655, 11420839, 11516071, 11716135, 11732839, 11742055, 11896423, 12043303, 12049447, 12075367, 12255271, 12454567, 12614503, 12645223, 12725287, 12893863, 13026535, 13099687, 13131943, 13231591, 13295143, 13321063, 13375015, 13595815, 13717543, 13769383, 13902823, 14028967, 14193511, 14287207, 14405479, 14411239, 14413543, 14425255, 14427559, 14436391, 14438119};

//	uint32_t comb[] = {0, 84480, 168960, 253440, 337920, 422400, 506880, 591360, 675840, 760320, 844800, 865920, 887040, 908160, 929280, 950400, 971520, 992640, 1013760, 1034880, 1056000, 1077120, 1098240, 1119360, 1140480, 1161600, 1182720, 1203840, 1224960, 1246080, 1267200, 1288320, 1309440, 1330560, 1351680, 1372800, 1393920, 1415040, 1520640, 1605120, 1689600, 1774080, 1858560, 1879680, 1900800, 1921920, 2027520, 2112000, 2196480, 2280960, 2365440, 2449920, 2534400, 2618880, 2703360, 2787840, 2872320, 2956800, 3041280, 3125760, 3210240, 3231360, 3252480, 3273600, 3294720, 3315840, 3336960, 3358080, 3379200, 3400320, 3421440, 3442560, 3463680, 3484800, 3505920, 3527040, 3548160};
//	uint32_t left[] = {4121674, 4135690, 4142026, 4144714, 4487818, 4704586, 4746826, 4899274, 4957642};

	//xcorr(toDense([0 1     3     4     5     7     9]), toDense([0,1,2,4,6]))
//	uint32_t left[] = {0, 1,     3,     4,     5,     7,     9};
//	uint32_t left[] = {2,3,5,6,7,9,11};
//	uint32_t comb[] = {0,1,2,4,6};
//	uint32_t left[] = {0,1,10};
//	uint32_t comb[] = {0,1,4};


	uint32_t answer;

	answer = pop_correlate(data, size, comb, ARRAY_LEN(comb));

	printf("\r\nMaxScore: %u\r\n", answer);





//     char c = data[0];
//
//     if( !headValid )
//     {
//             if( c == '\n' )
//                     headValid = true;
//     }
//     else
//     {
//
//             if( c == '\n' )
//             {
//                     parse();
//                     command.erase(command.begin(),command.end());
//             }
//             else
//             {
//                     command.push_back(c);
//             }
//     }
}
//
//
//unsigned parseHex(std::string &in)
//{
//     unsigned result;
//     std::stringstream ss;
//     ss << std::hex << in;
//     ss >> result;
//     return result;
//}
//
//unsigned parseInt(char *in)
//{
//     unsigned result;
//     std::stringstream ss;
//     ss << in;
//     ss >> result;
//     return result;
//}
//
//double parseDouble(std::string &in)
//{
//     double result;
//     std::stringstream ss;
//     ss << in;
//     ss >> result;
//     return result;
//}
//
//bool checksumOk(std::string &str, unsigned len)
//{
//     unsigned char givenChecksum;
//
//     if( len < 5 )
//             return false;
//
//     std::string checkStr = str.substr(len - 3, 2);
//
//     givenChecksum = parseHex(checkStr);
//
//
//     // this extracts bytes according to http://www.gpsinformation.org/dale/nmea.htm
//     std::string checkedBytes = str.substr(1, len - 5);
//
//     unsigned char calculatedChecksum = 0x00;
//
//     // loop and xor
//     for(char& c : checkedBytes) {
//             calculatedChecksum ^= c;
//     }
//
//     //      cout << "bytes: " << checkedBytes << endl;
//     //      cout << "checksum: " << (int) checksum << endl;
//     //      cout << "checksum bytes: " << checkStr << endl;
//     //      cout << "given: " << (int) givenChecksumInt << " calculated " << (int) calculatedChecksum << endl;
//
//
//     return (givenChecksum == calculatedChecksum);
//}
//
//double parseRetardedFormat(std::string &str, bool positive)
//{
//     size_t found;
//
//     found = str.find(".");
//
//     if( found == std::string::npos )
//     {
//             cout << "GPS format missing decimal" << endl;
//             return 0.0;
//     }
//
//     if( found < 2 )
//     {
//             cout << "LHS of GPS format is too small (" << found << ")" << endl;
//             return 0.0;
//     }
//
//     std::string minute = str.substr(found-2, str.length() - 1);
//     std::string degree = str.substr(0, found-2);
//
//     double sign = (positive)?1:-1;
//
////   cout << "Minute is: " << minute << " degree is: " << degree << endl;
//
//     return (parseDouble(degree) + (parseDouble(minute)/60)) * sign;
//}
//
//bool fixStatusOk(int status) {
//     if( status < 1 )
//             return false;
//
//     if( status > 6 )
//             return false;
//
//     return true;
//}
//
//void PopParseGPS::setFix(double lat, double lng, double time)
//{
//     boost::lock_guard<boost::mutex> guard(mtx_);
//     this->lat = lat;
//     this->lng = lng;
//     //time
//     this->gpsFix = true;
//}
//
//
//
//void PopParseGPS::gga(std::string &str)
//{
//     char seps[] = ",";
//     char *token;
//     char *state;
//     unsigned index = 0;
//     int fixStatus = -1;
//     std::string latStr, lngStr;
//     bool latPositive = true, lngPositive = true;
//
//     token = strtok_r_single( &str[0], seps, &state );
//     while( token != NULL )
//     {
//#ifdef POPPARSEGPS_VERBOSE
//             cout << index << ": " << token << endl;
//#endif
//
//             switch(index)
//             {
//                     case 2:
//                             latStr = std::string(token);
//                             break;
//                     case 3:
//                             latPositive = (strncmp("N", token, 1)==0)?true:false;
//                             break;
//                     case 4:
//                             lngStr = std::string(token);
//                             break;
//                     case 5:
//                             lngPositive = (strncmp("E", token, 1)==0)?true:false;
//                             break;
//                     case 6:
//                             fixStatus = parseInt(token);
//#ifdef POPPARSEGPS_VERBOSE
//                             cout << "Fix status: " << fixStatus << endl;
//#endif
//                             break;
//                     default:
//                             break;
//             }
//
//
//             token = strtok_r_single( NULL, seps, &state );
//             index++;
//     }
//
//     if( fixStatusOk(fixStatus) )
//     {
//             setFix(parseRetardedFormat(latStr, latPositive), parseRetardedFormat(lngStr, lngPositive), 0);
//     }
//
////   cout << boost::lexical_cast<string>( parseRetardedFormat(latStr, latPositive) ) << endl;
////   cout << boost::lexical_cast<string>( parseRetardedFormat(lngStr, lngPositive) )<< endl;
////   cout << "Fix ok: " << fixOk(fixStatus) << endl;
////   cout << latStr << latPositive << lngStr << lngPositive << endl;
//}
//
//bool PopParseGPS::gpsFixed()
//{
//     return gpsFix;
//}
//boost::tuple<double, double, double> PopParseGPS::getFix()
//{
//     boost::lock_guard<boost::mutex> guard(mtx_);
//     return boost::tuple<double, double, double>(this->lat, this->lng, 0.0);
//}
//
//void PopParseGPS::parse()
//{
//     std::size_t found;
//     bool cok = false;
//     unsigned len = command.size();
//     if( len == 0 )
//             return;
//
//     // str will contain the entire message, with a trailing \r (from \r\n)
//     std::string str(command.begin(),command.end());
//
//     cok = checksumOk(str, len);
//
//     if( !cok ) {
////           cout << "bad GPS checksum (" << str << ")" << endl;
//             return;
//     }
//
//
//#ifdef POPPARSEGPS_VERBOSE
//     cout << "command: " << str << endl;
//#endif
//
//     found = str.find("$GPGSA");
//     if( found == 0 )
//     {
//
//     }
//
//     found = str.find("$GPRMC");
//     if( found == 0 )
//     {
//
//     }
//
//     found = str.find("$GPGGA");
//     if( found == 0 )
//     {
//             gga(str);
//     }
//
//}


} //namespace

