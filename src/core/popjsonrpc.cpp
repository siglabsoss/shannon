#include "core/popjsonrpc.hpp"

#include <stddef.h>

#include <iostream>
#include <string>


using namespace std;



namespace pop
{



// functions
void PopJsonRPC::rcp_log(std::string log)
{
	cout << log << endl;
}


int rpc_count()
{
	static int i = 0;
	cout << "Bump to " << ++i << endl;
	return i;
}

void ppp(std::string p)
{
	cout << p << endl;
}


PopJsonRPC::PopJsonRPC(unsigned notused) : PopSink<char>("PopJsonRPCSink", 1), tx("PopJsonRPCResponse"), headValid(false), rawMode(false)
{
}

void PopJsonRPC::init() {}

void PopJsonRPC::process(const char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{
	if( size != 1 ) {
		cout << "Error " << this->get_name() << " may only accept 1 character at a time";
		return;
	}

	char c = data[0];

	//cout << c;

	// store characters until a null arrives
	// call parse once conditions are valid
	if( !headValid )
	{
		if( c == 0 )
			headValid = true;
	}
	else
	{

		if(!rawMode)
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
		else
		{
			execute_raw(c);
		}
	}
}


void PopJsonRPC::parse()
{
	unsigned len = command.size();
	if( len == 0 )
		return;

	std::string str(command.begin(),command.end());

	const char *json = str.c_str();

	struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS];
	const struct json_token *methodTok = 0, *paramsTok = 0, *resultTok = 0, *idTok = 0;

	int returnValue;
	int is_rpc = 1, is_result = 1;

	// Tokenize json string, fill in tokens array
	returnValue = parse_json(json, strlen(json), arr, POP_JSON_RPC_SUPPORTED_TOKENS);

	if( returnValue == JSON_STRING_INVALID || returnValue == JSON_STRING_INCOMPLETE )
	{
		// skip printing this message for simple newline messages.  if one string matches, it returns 0 which we then multiply
		if( ( str.compare("\r\n\r\n") * str.compare("\r\n") * str.compare("\n") * str.compare("\r") ) != 0)
		{
			cout << "problem with json string (" <<  str << ")" << endl;
		}
		return;
	}

	if( returnValue == JSON_TOKEN_ARRAY_TOO_SMALL )
	{
		cout << "problem with json string (too many things for us to parse)" << endl;
		return;
	}

	methodTok = find_json_token(arr, "method");
	if( !(methodTok && methodTok->type == JSON_TYPE_STRING) )
	{
		is_rpc = 0;
	}

	paramsTok = find_json_token(arr, "params");
	if( !(paramsTok && paramsTok->type == JSON_TYPE_ARRAY) )
	{
		is_rpc = 0;
	}

	resultTok = find_json_token(arr, "result");
	if( !resultTok )
	{
		is_result = 0;
	}

	// "id" key is optional.  It's absence means the message will not get a response
	idTok = find_json_token(arr, "id");
	if( !(idTok && idTok->type == JSON_TYPE_NUMBER) )
	{
		is_result = 0;
	}


	if( is_rpc )
	{
		execute_rpc(methodTok, paramsTok, idTok, arr, str);
	} else if( is_result )
	{
		execute_result(resultTok, idTok, arr, str);
	} else if( str.compare("{}") != 0 ) {
		// An "empty" response to a poll message is just an empty json object, everything else should generate this error
		cout << "Received valid json that doesn't look like JSON-RPC\r\n" << endl;
	}

}

void PopJsonRPC::send_rpc(std::string& rpc)
{
	send_rpc(rpc.c_str(), rpc.length());
}

void PopJsonRPC::send_rpc(const char *rpc_string, size_t length)
{
	// Leading null. Send this character as a precaution, in case the previous
	// RPC was not terminated properly. It's safe to do this because if Artemis
	// receives two null characters in a row, it will just ignore the empty RPC.
	this->tx.process("\0", 1);
	this->tx.process("\0", 1);

	this->tx.process(rpc_string, length);

	// Trailing null
	this->tx.process("\0", 1);
	this->tx.process("\0", 1);
}

uint16_t PopJsonRPC::rpc_get_autoinc(void)
{
	static uint16_t val = 1;
	return val++;
}



}
