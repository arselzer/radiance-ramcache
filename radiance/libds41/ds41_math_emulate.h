/* CPU-only test driver for the exact shared GPU row equations. This is not a serving row. */
#pragma once
#include "ds41_math_impl.h"
#include <vector>
namespace rad::ds41::math {
inline int emulate(Op op,const RadArgs* args) {
    Plan p;const int rc=validate(op,args,&p);if(rc!=RAD_OK)return rc;
    *p.status=RAD_OK;
    for(int64_t item=0;item<preflight_items(p);++item) {
        const int status=check_item(p,item);
        if(status!=RAD_OK) {*p.status=status;return status;}
    }
    std::vector<float> work(size_t(workspace_floats(p)));
    for(int64_t job=0;job<derived_jobs(p);++job) {
        const int status=checked_job<false>(p,job,work.data());
        if(status!=RAD_OK) {*p.status=status;return status;}
    }
    for(int64_t job=0;job<derived_jobs(p);++job) {
        const int status=checked_job<true>(p,job,work.data());
        if(status!=RAD_OK) {*p.status=status;return status;}
    }
    return RAD_OK;
}
} // namespace rad::ds41::math
