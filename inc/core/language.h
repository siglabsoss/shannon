
#ifndef __LANGUAGE_H
#define __LANGUAGE_H

// CMake defines __STDC_FORMAT_MACROS so that PRIuPTR will work.  An alternative is to just #define this before including
// See http://stackoverflow.com/questions/1403074/printf-with-sizeof-on-32-vs-64-platforms-how-do-i-handle-format-code-in-platfor
#include <inttypes.h>



#define _MSG_HEADER "%s - "

const char msg_out_of_memory[] = "out of memory";
const char msg_too_much_data[] = "received too much data from source";
const char msg_object_overflow[] = "%s overflow (%s did not process data quickly enough)\r\n"
                                   "\tYou can either 1) make the source buffer bigger or\r\n"
                                   "\t2) make the sink process data faster.";
const char msg_create_new_circ_buf[] = "\r\nCreating a new circular buffer for object %s\r\n";
const char msg_create_new_circ_buf_dbg_1[] = "data chunk = %" PRIuPTR " bytes, requested = %" PRIuPTR " bytes, ";
const char msg_create_new_circ_buf_dbg_2[] = "allocated = %" PRIuPTR " bytes";
const char msg_requested_mem_size[] = "Requested memSize=%" PRIuPTR ", actual=%" PRIuPTR ", number of samples=%" PRIuPTR "\r\n";
const char msg_no_buffer_allocated[] = _MSG_HEADER "No buffer has been allocated.";
const char msg_passing_invalid_amount_of_samples[] = _MSG_HEADER "Passing invalid amount of samples to class.";
const char msg_warning_replacing_gpu_sink[] = "\r\nWarning: Replacing existing sink for object%s\r\n";



const char msg_need_more_than_zero_channels[] = "need to initialize class with more than zero channels";

#define BLACK "\033[22;30m"
#define RED "\033[22;31m"
#define GREEN "\033[22;32m"
#define YELLOW "\033[22;33m"
#define BLUE "\033[22;34m"
#define MAGENTA "\033[22;35m"
#define CYAN "\033[22;36m"
#define WHITE "\033[22;37m"
#define RESETCOLOR "\033[0m"

#endif // __LANGUAGE_H
