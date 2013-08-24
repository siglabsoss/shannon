server.out: client.out async_udp_echo_server.cpp
	g++ async_udp_echo_server.cpp -o server.out -lpthread -lboost_system
client.out: blocking_udp_echo_client.cpp
	g++ blocking_udp_echo_client.cpp -o client.out -lpthread -lboost_system
clean:
	rm server.out client.out
