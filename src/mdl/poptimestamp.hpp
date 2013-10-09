#ifndef __POP_TIMESTAMP_HPP_
#define __POP_TIMESTAMP_HPP_

#include <time.h>
#include <boost/operators.hpp>
#include <uhd/types/time_spec.hpp>


namespace pop
{

class PopTimestamp
{
	/*!
	 * A time_spec_t holds a seconds and a fractional seconds time value.
	 * Depending upon usage, the time_spec_t can represent absolute times,
	 * relative times, or time differences (between absolute times).
	 *
	 * The time_spec_t provides clock-domain independent time storage,
	 * but can convert fractional seconds to/from clock-domain specific units.
	 *
	 * The fractional seconds are stored as double precision floating point.
	 * This gives the fractional seconds enough precision to unambiguously
	 * specify a clock-tick/sample-count up to rates of several petahertz.
	 */
	//	class UHD_API time_spec_t : boost::additive<time_spec_t>, boost::totally_ordered<time_spec_t>{
public:

	/*!
	 * Copy constructor from time_spec_t type
	 */
	PopTimestamp(uhd::time_spec_t copy, double off) : _full_secs(copy.get_full_secs()), _frac_secs(copy.get_frac_secs()), offset(off) {}

	/*!
	 * Copy constructor from PopTimestamp with explicit setter for new offset
	 */
	PopTimestamp(PopTimestamp copy, double off) : _full_secs(copy.get_full_secs()), _frac_secs(copy.get_frac_secs()), offset(off) {}

	/*!
	 * Get the system time in time_spec_t format.
	 * Uses the highest precision clock available.
	 * \return the system time as a time_spec_t
	 */
	static PopTimestamp get_system_time(void);

//	/*!
//	 * Create a time_spec_t from a real-valued seconds count.
//	 * \param secs the real-valued seconds count (default = 0)
//	 */
	PopTimestamp(double secs = 0);
//
//	/*!
//	 * Create a time_spec_t from whole and fractional seconds.
//	 * \param full_secs the whole/integer seconds count
//	 * \param frac_secs the fractional seconds count (default = 0)
//	 */
	PopTimestamp(time_t full_secs, double frac_secs = 0);

	/*!
	 * Create a time_spec_t from whole seconds and fractional ticks.
	 * Translation from clock-domain specific units.
	 * \param full_secs the whole/integer seconds count
	 * \param tick_count the fractional seconds tick count
	 * \param tick_rate the number of ticks per second
	 */
	PopTimestamp(time_t full_secs, long tick_count, double tick_rate);
//
//	/*!
//	 * Create a time_spec_t from a 64-bit tick count.
//	 * Translation from clock-domain specific units.
//	 * \param ticks an integer count of ticks
//	 * \param tick_rate the number of ticks per second
//	 */
//	static time_spec_t from_ticks(long long ticks, double tick_rate);
//
//	/*!
//	 * Convert the fractional seconds to clock ticks.
//	 * Translation into clock-domain specific units.
//	 * \param tick_rate the number of ticks per second
//	 * \return the fractional seconds tick count
//	 */
//	long get_tick_count(double tick_rate) const;
//
//	/*!
//	 * Convert the time spec into a 64-bit tick count.
//	 * Translation into clock-domain specific units.
//	 * \param tick_rate the number of ticks per second
//	 * \return an integer number of ticks
//	 */
//	long long to_ticks(const double tick_rate) const;
//
//	/*!
//	 * Get the time as a real-valued seconds count.
//	 * Note: If this time_spec_t represents an absolute time,
//	 * the precision of the fractional seconds may be lost.
//	 * \return the real-valued seconds
//	 */
	double get_real_secs(void) const;
//
//	/*!
//	 * Get the whole/integer part of the time in seconds.
//	 * \return the whole/integer seconds
//	 */
//	time_t get_full_secs(void) const;
//
//	/*!
//	 * Get the fractional part of the time in seconds.
//	 * \return the fractional seconds
//	 */
//	double get_frac_secs(void) const;
//



	//! Add double to timestamp
	PopTimestamp &operator+=(const double &rhs);

	//! Implement addable interface
	PopTimestamp &operator+=(const PopTimestamp &);

	//! Implement subtractable interface
	PopTimestamp &operator-=(const PopTimestamp &);

	//public time storage details

	// time stuff
	time_t _full_secs;
	double _frac_secs;

	// which sample does this apply to
	double offset;

	//	//! Implement equality_comparable interface
	//	UHD_API bool operator==(const time_spec_t &, const time_spec_t &);
	//
	//	//! Implement less_than_comparable interface
	//	UHD_API bool operator<(const time_spec_t &, const time_spec_t &);
	//
	time_t get_full_secs(void) const{
		return _full_secs;
	}

	double get_frac_secs(void) const{
		return _frac_secs;
	}

	double offset_adjusted(double o) const{
		return offset - o;
	}



}; // class PopTimestamp

extern bool operator==(const PopTimestamp &lhs, const PopTimestamp &rhs);



} // namespace pop



#endif
