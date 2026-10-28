/* The dispatch-log globals r4d_emit.h references, normally defined in r4d_rows.cpp, so a single
 * libr4d unit can be built into its own shared object for ds41_mxfp4_device_check. */
const void** r4d_dispatch_log = nullptr;
int          r4d_dispatch_cap = 0;
int          r4d_dispatch_n = 0;
