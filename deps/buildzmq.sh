#!/bin/bash

echo "You must \`sudo apt-get remove libzmq-dev; sudo apt-get autoremove\` before running this"

sleep 2

# Header
sudo cp cppzmq/zmq.hpp  /usr/include/
sudo chown root. /usr/include/zmq.hpp
chmod 644 /usr/include/zmq.hpp

tar -zxf zeromq-4.0.4.tar.gz
cd zeromq-4.0.4
./configure
make -j2
sudo make install

