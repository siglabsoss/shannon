#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <map>
#include <vector>
#include <boost/variant.hpp>
#include "json/json.h"

namespace rbx
{
/// this class has its own internal singleton instead of using the Singleton
/// class to avoid using the syntax Config::get()->get() which is pretty ugly
class Config
{
public:
	/// load a config file into the specified group
	static void loadFile( std::string const& filename );

	template<class T>
	static T get( std::string const& key );

	/// public destructor
	~Config() {}

private:
	Config() : mJsonStorage(0) {}

	/// create an instance of the class
	static void init()
	{
		if(!instance)
			instance = new Config();
	}

	/// a pointer to an instance of this class, for singleton purposes
	static Config *instance;

	/// holds data
	json::value mJsonStorage;
};

}

#endif
