#include "core/popjsonrpc.hpp"

#include <iostream>
#include <string>


using namespace std;



namespace pop
{



// functions
void rcp_log(std::string log)
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


PopJsonRPC::PopJsonRPC(unsigned notused) : PopSink<unsigned char>("PopJsonRPCSink", 1), rx("PopJsonRPCResponse"), headValid(false)
{
	tx = this;
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
	const struct json_token *methodTok, *paramsTok, *idTok;

	// Tokenize json string, fill in tokens array
	int returnValue = parse_json(json, strlen(json), arr, POP_JSON_RPC_SUPPORTED_TOKENS);

	if( returnValue == JSON_STRING_INVALID || returnValue == JSON_STRING_INCOMPLETE )
	{
		cout << "problem with json string (" <<  str << ")" << endl;
		return;
	}

	if( returnValue == JSON_TOKEN_ARRAY_TOO_SMALL )
	{
		cout << "problem with json string (too many things for us to parse)" << endl;
		return;
	}

//	std::string method;
	// Search for parameter "bar" and print it's value
	methodTok = find_json_token(arr, "method");
	if( !(methodTok && methodTok->type == JSON_TYPE_STRING) )
	{
		return;
	}
	else
	{
//		method = std::string(methodTok->ptr, methodTok->len);
	}


	paramsTok = find_json_token(arr, "params");
	if( !(paramsTok && paramsTok->type == JSON_TYPE_ARRAY) )
	{
		return;
	}

	int methodId = -1;
	idTok = find_json_token(arr, "id");
	if( !(idTok && idTok->type == JSON_TYPE_NUMBER) )
	{
//		return;
	}
	else
	{
//		std::string sval = std::string(tok3->ptr, tok3->len);
//		methodId = std::stoi(sval);
//
//		if( methodId < 0 )
//			return;
	}




	execute(methodTok, paramsTok, idTok, arr, str);

}

void PopJsonRPC::execute(const struct json_token *methodTok, const struct json_token *paramsTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], std::string str)
{
	std::string method = std::string(methodTok->ptr, methodTok->len);
	const struct json_token *p0, *p1, *p2;

	if( method.compare("log") == 0 )
	{
		p0 = find_json_token(arr, "params[0]");
		if( p0 && p0->type == JSON_TYPE_STRING )
		{
			rcp_log(std::string(p0->ptr, p0->len));
//			respond_int(0, methodId);
		}
	}

//	if( method.compare("count") == 0 )
//	{
//		int ret = rpc_count();
//		respond_int(ret, methodId);
//	}

	if( method.compare("rx") == 0 )
	{
		p0 = find_json_token(arr, "params[0]");
		if( p0 && p0->type == JSON_TYPE_STRING )
		{
			cout << "got rx" << endl;
//			rcp_log(std::string(tok->ptr, tok->len));
			//			respond_int(0, methodId);
		}
	}
}

void PopJsonRPC::respond_int(int value, int methodId)
{

	std::ostringstream ss;
	//    	ss << "This is " << cs << "!";
	//    	std::cout << ss.str() << std::endl

	ss << "{\"result\":" << value << ", \"error\": null, \"id\": " << methodId << "}";

	std::string str = ss.str();
	unsigned char *buff;

	buff = rx.get_buffer(1);
	buff[0] = '\0';
	rx.process(1);

	// should copy in all the characters but omit the final null
	buff = rx.get_buffer(str.size());
	strncpy((char*)buff, str.c_str(), str.size());
	rx.process(str.size());

	buff = rx.get_buffer(1);
	buff[0] = '\0';
	rx.process(1);
}

}
