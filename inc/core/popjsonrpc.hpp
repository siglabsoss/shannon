#ifndef __POP_JSON_RPC_HPP_
#define __POP_JSON_RPC_HPP_


#include <core/popsink.hpp>
#include <core/popsource.hpp>
#include <frozen/frozen.h>

using namespace std;

#define POP_JSON_RPC_SUPPORTED_TOKENS (10)

namespace pop
{



// functions
void rcp_log(std::string log)
{
	cout << log << endl;
}


void rpc_count()
{
	static int i = 0;
	cout << "Bump to " << i++ << endl;
}

void ppp(std::string p)
{
	cout << p << endl;
}











class PopJsonRPC : public PopSink<unsigned char>
{
public:
	 PopSink<unsigned char> *tx; // PopSink must be inherited b/c of virtual classes, so we fake out a pointer here
	 bool headValid;
	 std::vector<unsigned char> command;

private:
	// pointer to a single timestamp given to us in the previous call to process
//	const PopTimestamp* previous_timestamp;
//
//	// the number of data samples from the previous call to process
//	size_t previous_size;
public:
	PopJsonRPC(size_t chunk) : PopSink<unsigned char>("PopJsonRPCSink", 1), headValid(false)
    {
		 tx = this;
    }
    void init() {}
    void process(const unsigned char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
    {
    	if( size != 1 ) {
    		cout << "Error " << this->get_name() << " may only accept 1 character at a time";
    		return;
    	}

    	char c = data[0];

    	if( !headValid )
    	{
    		if( c == 0 )
    			headValid = true;
    	}
    	else
    	{

    		if( c == 0 )
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

    void parse()
    {
    	unsigned len = command.size();
    	if( len == 0 )
    		return;

    	std::string str(command.begin(),command.end());

    	const char *json = str.c_str();

    	struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS];
    	const struct json_token *tok, *tok2;

    	// Tokenize json string, fill in tokens array
    	int returnValue = parse_json(json, strlen(json), arr, POP_JSON_RPC_SUPPORTED_TOKENS);

    	if( returnValue == JSON_STRING_INVALID || returnValue == JSON_STRING_INCOMPLETE )
    	{
    		cout << "problem with json string" << endl;
    		return;
    	}

    	if( returnValue == JSON_TOKEN_ARRAY_TOO_SMALL )
    	{
    		cout << "problem with json string (too many things for us to parse)" << endl;
    		return;
    	}

    	std::string method;
    	// Search for parameter "bar" and print it's value
    	tok = find_json_token(arr, "method");
    	if( !(tok && tok->type == JSON_TYPE_STRING) )
    	{
    		return;
//
//    		cout << "got : " << method << endl;
    	}
    	method = std::string(tok->ptr, tok->len);

    	tok2 = find_json_token(arr, "params");

    	if( !(tok2 && tok2->type == JSON_TYPE_ARRAY) )
		{
    		return;
//			std::string params = std::string(tok2->ptr, tok2->len);
//
//			cout << "got : " << params << endl;
		}
//    	cout << tok->type << endl;
//    	printf("Value of bar is: [%.*s]\n", tok->len, tok->ptr);

    	execute(method, arr);

    }

    void execute(std::string &method, json_token *tokens)
    {
     	const struct json_token *tok;
    	if( method.compare("log") == 0 )
    	{
    		tok = find_json_token(tokens, "params[0]");
    		if( tok && tok->type == JSON_TYPE_STRING )
    		{
    			rcp_log(std::string(tok->ptr, tok->len));
    		}
    	}

    	if( method.compare("count") == 0 )
    	{
    		rpc_count();
    	}
    }
};

}


#endif
