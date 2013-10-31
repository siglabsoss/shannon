#ifndef __PHYMESSAGE_H__
#define __PHYMESSAGE_H__


typedef struct PHY_MSG_T
{
	uint8_t type;
	uint8_t id[6];
	uint8_t len;
} PHY_MSG;

typedef struct PHY_MSG_HELO_T
{
	PHY_MSG header;
	uint8_t bat;
	uint8_t temp;
} PHY_MSG_HELO;

typedef enum PHY_MSG_TYPE_T
{
	PHY_MSG_HELO_TYPE = 1
} PHY_MSG_TYPE;



#endif
