#include "core/popjsonrpc.hpp"

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


PopJsonRPC::PopJsonRPC(unsigned notused) : PopSink<unsigned char>("PopJsonRPCSink", 1), tx("PopJsonRPCResponse"), headValid(false)
{
}

void PopJsonRPC::init() {}

void PopJsonRPC::process(const unsigned char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
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


void PopJsonRPC::parse()
{
	unsigned len = command.size();
	if( len == 0 )
		return;

	std::string str(command.begin(),command.end());

	const char *json = str.c_str();

	struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS];
	const struct json_token *methodTok = 0, *paramsTok = 0, *idTok = 0;

	// Tokenize json string, fill in tokens array
	int returnValue = parse_json(json, strlen(json), arr, POP_JSON_RPC_SUPPORTED_TOKENS);

	if( returnValue == JSON_STRING_INVALID || returnValue == JSON_STRING_INCOMPLETE )
	{
		// skip printing this message for simple newline messages.  if one string matches, it returns 0 which we then multiply
		if( ( str.compare("\r\n") * str.compare("\n") * str.compare("\r") ) != 0)
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

	// verify message has "method" key
	methodTok = find_json_token(arr, "method");
	if( !(methodTok && methodTok->type == JSON_TYPE_STRING) )
	{
		return;
	}

	// verify message has "params" key
	paramsTok = find_json_token(arr, "params");
	if( !(paramsTok && paramsTok->type == JSON_TYPE_ARRAY) )
	{
		return;
	}

	// "id" key is optional.  It's absense means the message will not get a response

	execute(methodTok, paramsTok, idTok, arr, str);
}

//TODO: flush out implementation and use this
void PopJsonRPC::respond_int(int value, int methodId)
{
	std::ostringstream ss;

	ss << "{\"result\":" << value << ", \"error\": null, \"id\": " << methodId << "}";

	std::string str = ss.str();
	unsigned char *buff;

	buff = tx.get_buffer(1);
	buff[0] = '\0';
	tx.process(1);

	// should copy in all the characters but omit the final null
	buff = tx.get_buffer(str.size());
	strncpy((char*)buff, str.c_str(), str.size());
	tx.process(str.size());

	buff = tx.get_buffer(1);
	buff[0] = '\0';
	tx.process(1);
}


}
