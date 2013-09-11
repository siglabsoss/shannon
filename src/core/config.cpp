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
bool Config::get<bool>( std::string const& key )
{
	init();
	return json::to_bool(instance->mJsonStorage[key]);
}
}
