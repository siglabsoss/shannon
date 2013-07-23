


const char msg_out_of_memory[] = "out of memory";
const char msg_passing_invalid_amount_of_samples[] = "passing invalid amount of samples to class";
const char msg_too_much_data[] = "received too much data from source";
const char msg_page_div_into_data[] = "currently page size must be evenly divisible by data size";
const char msg_object_overflow[] = "%s overflow (%s did not process data quickly enough)\r\n"
                                   "\tYou can either 1) make the source buffer bigger or\r\n"
                                   "\t2) make the sink process data faster.";
const char msg_create_new_circ_buf[] = "\r\nCreating a new circular buffer for object %s\r\n";
const char msg_requested_mem_size[] = "Requested memSize=%lu, actual=%lu, number of samples=%lu\r\n";



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
