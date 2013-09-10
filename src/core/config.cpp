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



  json::value json = json::parse(first, last);

//  ConfigGrammar< file_iterator<>, qi::rule<file_iterator<> > > parser;
//
//  ConfigTree config;
  bool parseSucceeded = false;
  try
    {
	  instance->mJsonStorage = json::parse(first, last);

//	  cout << json::to_string(instance->mJsonStorage["key"]) << endl;

//      parseSucceeded = qi::phrase_parse(first, last, parser, parser.getCommentRule(), config);
    }
  catch (std::exception const& x)
    {
      std::ostringstream error;
      error << "expected: " << x.what();
//      error << ", but got: \"" << std::string(x.first, x.last) << '"' << std::endl;
      cout << "Parsing " << error;
    }
//
//  if( parseSucceeded )
//    {
//      for( std::vector<ConfigVariable>::const_iterator it = config.data.begin();
//	   it != config.data.end();
//	   it++ )
//	{
//	  instance->set( config.group, it->variableName, it->variableData );
//	  //instance->mVariables[config.group][it->variableName] = it->variableData;
//	}
//    }
}

namespace rbx // template specializations need to be in the cpp file but in the same namespace
{
  /*
  template<>
  std::vector<int> Config::get< std::vector<int> >( std::string const& group, std::string const& var )
  {
    init();
    
    return instance->mVariables[group][var];
  }
  */
//
//
//
//  template<>
//  void Config::set<int>( std::string const& group, std::string const& var, int const& val )
//  {
////    init();
////    //instance->mVariables[group][var].front() = val;
////    instance->mVariables[group][var] = val;
//  }

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

//Config::Config( std::string const& group )
//{
//  init();
//  mGroup = group;
//}
