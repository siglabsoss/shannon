#include "net/popwebhook.hpp"
//#include <iostream>
#include <boost/asio.hpp>
#include "core/config.hpp"
//#include <string>

using namespace std;



// http://stackoverflow.com/questions/2616011/easy-way-to-parse-a-url-in-c-cross-platform
struct Uri
{
public:
std::string QueryString, Path, Protocol, Host, Port;

static Uri Parse(const std::string &uri)
{
    Uri result;

    typedef std::string::const_iterator iterator_t;

    if (uri.length() == 0)
        return result;

    iterator_t uriEnd = uri.end();

    // get query start
    iterator_t queryStart = std::find(uri.begin(), uriEnd, '?');

    // protocol
    iterator_t protocolStart = uri.begin();
    iterator_t protocolEnd = std::find(protocolStart, uriEnd, ':');            //"://");

    if (protocolEnd != uriEnd)
    {
        std::string prot = &*(protocolEnd);
        if ((prot.length() > 3) && (prot.substr(0, 3) == "://"))
        {
            result.Protocol = std::string(protocolStart, protocolEnd);
            protocolEnd += 3;   //      ://
        }
        else
            protocolEnd = uri.begin();  // no protocol
    }
    else
        protocolEnd = uri.begin();  // no protocol

    // host
    iterator_t hostStart = protocolEnd;
    iterator_t pathStart = std::find(hostStart, uriEnd, '/');  // get pathStart

    iterator_t hostEnd = std::find(protocolEnd,
        (pathStart != uriEnd) ? pathStart : queryStart,
        L':');  // check for port

    result.Host = std::string(hostStart, hostEnd);

    // port
    if ((hostEnd != uriEnd) && ((&*(hostEnd))[0] == ':'))  // we have a port
    {
        hostEnd++;
        iterator_t portEnd = (pathStart != uriEnd) ? pathStart : queryStart;
        result.Port = std::string(hostEnd, portEnd);
    }

    // path
    if (pathStart != uriEnd)
        result.Path = std::string(pathStart, queryStart);

    // query
    if (queryStart != uriEnd)
        result.QueryString = std::string(queryStart, uri.end());

    return result;

}   // Parse
};  // uri





namespace pop
{

PopWebhook::PopWebhook(unsigned notused) : PopSink<PopRadio>("popwebhook", 1)
{

}

void PopWebhook::process(const PopRadio* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{
	const PopRadio *r = &(data[0]);

	std::stringstream buildBody;
	buildBody << "{\"updates\":[" << r->seralize() << "]}";
	std::string body = buildBody.str();

	cout << body << endl;


	doHook(Config::get<std::string>("s3p_webhook_url"), body);
}

void PopWebhook::doHook(std::string urlString, std::string body)
{
	Uri uri = Uri::Parse(urlString);

	cout << uri.QueryString << endl;
	cout << uri.Path << endl;
	cout << uri.Protocol << endl;
	cout << uri.Host << endl;
	cout << uri.Port << endl;

	// required to handle urls without a port
	std::string port = uri.Port.empty()?"80":uri.Port;


	boost::asio::ip::tcp::iostream stream;
	stream.expires_from_now(boost::posix_time::seconds(60));
	stream.connect(uri.Host,port);
	stream << "POST " << uri.Path << " HTTP/1.1\r\n";
	stream << "Accept: */*\r\n";
	stream << "Connection: close\r\n";
	stream << "Host: " << uri.Host << "\r\n";
	stream << "User-Agent: PopWi Webhook\r\n";
	stream << "Content-Type: application/json\r\n";
	stream << "Content-Length: " << body.length() << "\r\n";
	stream << "\r\n";
	stream << body;
	stream.flush();

	boost::posix_time::milliseconds workTime(1000);
	boost::this_thread::sleep(workTime);


}

} //namespace

