#ifndef __POP_RADIO_H_
#define __POP_RADIO_H_

#include <string>

namespace pop
{

class PopRadio
{
public:
	PopRadio();
	~PopRadio();
	float getBatCurrent() const;
	void setBatCurrent(float batCurrent);
	float getBatVoltage() const;
	void setBatVoltage(float batVoltage);
	double getLat() const;
	void setLat(double lat);
	double getLng() const;
	void setLng(double lon);
	long getSerial() const;
	void setSerial(long serial);
	int getStatus() const;
	void setStatus(int status);
	float getTemp() const;
	void setTemp(float temp);

	std::string seralize();

private:
	double lat;
	double lng;
	float bat_current;
	float bat_voltage;
	long serial;
	float temp;
	int status;
};



} // namespace pop

#endif
