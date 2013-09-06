/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_ASSERT_H_
#define __POP_ASSERT_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef enum POP_ERROR_T
{
	POP_ERROR_NONE,
	POP_ERROR_UNKNOWN,
	POP_ERROR_HW,
	POP_ERROR_ALREADY_RUNNING,
} POP_ERROR;

extern char *g_rgszPopError[];


inline static void _PopPrintCompilerInfo(void)
{
#ifdef __GNUC__
	printf("source file modified on %s\r\n", __TIMESTAMP__);
	printf("compiled on %s %s\r\n", __DATE__, __TIME__);
	printf("GNU Version %d.%d.%d\r\n", __GNUC__, __GNUC_MINOR__,
		__GNUC_PATCHLEVEL__);

#ifdef __STRICT_ANSI__
	printf("strict ANSI (-ansi or -std=c++98)\r\n");
#endif // __STRICT_ANSI__

#endif // __GNUC__
}


inline static POP_ERROR _PopCheckError(POP_ERROR error, const char* str,
	const char* func, const char* file,
	int line)
{

	if( POP_ERROR_NONE != error )
	{
		printf("Error at %s in function %s, file %s, line %d - %s(%d)\r\n",
			str, func, file, line, g_rgszPopError[error], error);
		_PopPrintCompilerInfo();

#ifdef DEBUG
		abort();
#endif // DEBUG

	}

	return error;
}


inline static void _PopAssert(int expression, const char* str,
	const char* func, const char* file, int line)
{
	if( 0 == expression )
	{
		printf("Assertion failed: %s in function %s, file %s, line %d\r\n",
			str, func, file, line);
		_PopPrintCompilerInfo();

#ifdef DEBUG
		abort();
#endif // DEBUG

	}
}


#define PopAssert(x) _PopAssert(x, #x, __func__, __FILE__, __LINE__)
#define PopAssertMessage(x, message) _PopAssert(x, #x " (" message ")", __func__, __FILE__, __LINE__)
#define PopCheckError(x) _PopCheckError(x, #x, __func__, __FILE__, __LINE__)


#endif // __POP_ASSERT_H_
