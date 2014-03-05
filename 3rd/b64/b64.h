/*
 * b64.h
 *
 *  Created on: Mar 3, 2014
 *      Author: joel
 */

#ifndef __B64_H__
#define __B64_H__

int b64_encode( const char *inbytes, unsigned count, char *outbytes, unsigned *countOut );
int b64_decode( const char *inbytes, unsigned count, char *outbytes, unsigned *countOut );
unsigned b64_length_encoded(unsigned len);

#endif /* B64_H_ */
