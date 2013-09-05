/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_EXCEPTION_H
#define __POP_EXCEPTION_H

#include <stdexcept>
#include <cstdarg>
#include <cstdio>
#include <cstring>


namespace pop
{
	class PopException : public std::exception
	{
	public:
		PopException(const char *format, ...)
		{
			va_list args;
			size_t nSize;

			va_start( args, format );

			nSize = vsnprintf( 0, 0, format, args );

			m_pszMsg = (char*)malloc( nSize + 3 /* EOL */);

			if( m_pszMsg )
			{
				va_start( args, format );

				vsnprintf( m_pszMsg, nSize + 1, format, args );

				strcat( m_pszMsg, "\r\n" );
			}

		}

		~PopException() throw()
		{
			if( m_pszMsg ) free( m_pszMsg );
		}

		const char *what() const throw()
		{
			if( m_pszMsg )
				return m_pszMsg;
			else
				return "[PopException] - exception class out of memory\r\n";
		}

	private:
		char *m_pszMsg;
	};

} // namespace pop

#endif // __POP_EXCEPTION_H
