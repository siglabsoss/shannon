/**
 * @file Config.cpp
 * @author Kito Berg-Taylor
 * @author Brinton Engineering LLC
 * @author Robotex Inc.
 *
 * Class for handling all config files and configuration options
 *
 */

#include "core/config.hpp"
#include <fstream>
#include <boost/spirit/include/classic_file_iterator.hpp>
#include <boost/spirit/include/qi.hpp>


using namespace rbx;
using namespace std;
using boost::spirit::classic::file_iterator;
namespace qi = boost::spirit::qi;

rbx::Config* rbx::Config::instance = 0;

void Config::loadFile( std::string const& filename )
{
	init();

	file_iterator<> first( filename.c_str() ), last;

	if (!first)
		cout << "Unable to open the config file: " << filename << " (or empty) "<< endl;

	last = first.make_end();

	bool parseSucceeded = false;
	try
	{
		instance->mJsonStorage = json::parse(first, last);
	}
	catch (std::exception const& x)
	{
		std::ostringstream error;
		error << "expected: " << x.what();
		cout << "Parsing " << error;
	}
}

void Config::loadFromDisk()
{
	rbx::Config::loadFile( DEFAULT_CONFIG_FILE_PATH );
}

namespace rbx // template specializations need to be in the cpp file but in the same namespace
{

// see json.tcc if types need to be added

template<>
std::string Config::get<std::string>( std::string const& key )
{
	init();
	return json::to_string(instance->mJsonStorage[key]);
}

template<>
double Config::get<double>( std::string const& key )
{
	init();
	return json::to_number(instance->mJsonStorage[key]);
}

template<>
int Config::get<int>( std::string const& key )
{
	init();
	return (int)json::to_number(instance->mJsonStorage[key]);
}

template<>
bool Config::get<bool>( std::string const& key )
{
	init();
	return json::to_bool(instance->mJsonStorage[key]);
}
}










#ifdef UNIT_TEST

BOOST_AUTO_TEST_SUITE( config_class )

BOOST_AUTO_TEST_CASE( init )
{
	ofstream myJsonFile;
	myJsonFile.open ("init_test.json.txt");
	myJsonFile << "{\"key\":\"value\",\"date\":234234}" << endl;
	myJsonFile.close();



	rbx::Config::loadFile( "init_test.json.txt" );


	std::string d = "date";
	double date = Config::get<double>(d);

	BOOST_CHECK_EQUAL( date, 234234 );

	double date2 = Config::get<double>("date");

	// redudnant? checking if can use std::string and c-string to fetch keys
	BOOST_CHECK_EQUAL( date, date2 );

	BOOST_CHECK_EQUAL( "value", Config::get<string>("key") );


	BOOST_CHECK_THROW (Config::get<string>("key doesn't exist so it should throw"), std::exception);



}


BOOST_AUTO_TEST_SUITE_END()

#endif
