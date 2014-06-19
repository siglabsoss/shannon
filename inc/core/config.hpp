#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <map>
#include <vector>
#include <boost/variant.hpp>
#include "json/json.h"

#define DEFAULT_CONFIG_FILE_PATH "node_config.json"

namespace pop
{
/// this class has its own internal singleton instead of using the Singleton
/// class to avoid using the syntax Config::get()->get() which is pretty ugly
class Config
{
public:
	/// load a config file into the specified group
	static void loadFile( std::string const& filename );

	// load the default config for shannon or s3p or whatever we are
	static void loadFromDisk( void );

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
