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

    /// set a config variable in the given group
//    template<class T>
//    static void set( std::string const& group, std::string const& var, T const& val );

    /// get a config variable from the given group
//    template<class T>
//    static T get( std::string const& group, std::string const& var );

    template<class T>
    static T get( std::string const& key );

//    template<class T>
//    static T get( std::string key );

//    template<class T>
//    T get( const char* key );

    /// create an instance of the class tied to a specific group
//    Config( std::string const& group );

    /// public destructor
    ~Config() {}

    /// set a config variable
    /// this should only be used to set single variables and not vectors
//    template<class T>
//    void set( std::string const& var, T const& val );
//
//    /// get a config variable
//    template<class T>
//    T get( std::string const & var ) const;

  private:
    Config() : mJsonStorage(0) {}
    Config& operator=( Config const& rhs ) const{}

    /// create an instance of the class
    static void init()
    {
    	if(!instance)
    		instance = new Config();
    }

    /// a pointer to an instance of this class, for singleton purposes
    static Config *instance;

    /// all the actual variables
    //std::map<std::string, std::map<std::string, std::vector<boost::variant<int, double, std::string> > > >  mVariables;
    /// TODO: change to boost::any to accomodate the hardware variables
    std::map<std::string, std::map<std::string, boost::variant<int, double, std::string, bool> > >  mVariables;

    /// the group that this object is tied to
    std::string mGroup;

    /// holds data
    json::value mJsonStorage;
  };


  



//
//  template<class T>
//  T get( const char* key )
//  {
//	  return get<T>(std::string(key));
//  }



//  template<class T>
//  T Config::get( std::string const& group, std::string const& var )
//  {
//    init();
//
//    if( instance->mVariables.count(group) != 1 )
//      std::cout << "Group does not exist: " << group << std::endl;
//
//    if( instance->mVariables[group].count(var) != 1 )
//    	std::cout << "Variable " << var << " does not exist in group " << std::endl;
//
//    //return instance->mVariables[group][var].front();
//    return boost::get<T>(instance->mVariables[group][var]);
//  }

//  template< class T >
//  void Config::set( std::string const& var, T const& val )
//  {
//    set(mGroup, var, val);
//  }

//  template< class T >
//  T Config::get( std::string const& var ) const
//  {
//    return get<T>( mGroup, var );
//  }


}

#endif
