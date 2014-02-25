#include <core/popgravitinoparser.hpp>


using namespace std;

#include <frozen/frozen.h>

namespace pop
{


long parseLong(const std::string &in)
{
	long result;
	std::stringstream ss;
	ss << in;
	ss >> result;
	return result;
}

double parseDouble(const std::string &in)
{
	double result;
	std::stringstream ss;
	ss << in;
	ss >> result;
	return result;
}

PopGravitinoParser::PopGravitinoParser() : PopSink<char>( "PopGravitinoParser", 1 ), headValid(false)
{
}

void PopGravitinoParser::init()
{
}

void PopGravitinoParser::process(const char* data, size_t data_size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{
	if( data_size != 1 ) {
		cout << "Error " << this->get_name() << " may only accept 1 character at a time" << endl;
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

void PopGravitinoParser::parse()
{
	unsigned len = command.size();
	if( len == 0 )
		return;

	std::string str(command.begin(),command.end());

	cout << str << endl;

	const char *json = str.c_str();

	struct json_token arr[POP_GRAVITINO_SUPPORTED_TOKENS];
	const struct json_token *tok, *tok2, *tok3;

	// Tokenize json string, fill in tokens array
	int returnValue = parse_json(json, strlen(json), arr, POP_GRAVITINO_SUPPORTED_TOKENS);

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



	long serial;
	double lat,lng;

	std::string method, serialString;

	tok = find_json_token(arr, "serial");
	if( !(tok && tok->type == JSON_TYPE_NUMBER) )
	{
		return;
	}
	else
	{
		serial = parseLong(std::string(tok->ptr, tok->len));
	}

	tok = find_json_token(arr, "lat");
	if( !(tok && tok->type == JSON_TYPE_NUMBER) )
	{
		return;
	}
	else
	{
		lat = parseDouble(std::string(tok->ptr, tok->len));
	}

	tok = find_json_token(arr, "lng");
	if( !(tok && tok->type == JSON_TYPE_NUMBER) )
	{
		return;
	}
	else
	{
		lng = parseDouble(std::string(tok->ptr, tok->len));
	}


	PopRadio *r = radios[serial];
	r->setLat(lat);
	r->setLng(lng);

	cout << "built object: " << r->seralize() << endl;

}

} //namespace

