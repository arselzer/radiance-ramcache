#pragma once
#include "ds41_engram_gate_impl.h"
namespace rad::ds41::engram_gate {
inline int emulate(const RadArgs* args) {
    Plan p;int rc=validate(args,&p);if(rc!=RAD_OK)return rc;
    *p.status=RAD_OK;
    for(int64_t i=0;i<preflight_items(p);++i) {
        rc=check_item(p,i);if(rc!=RAD_OK) {*p.status=rc;return rc;}
    }
    const int64_t jobs=p.tokens*p.copies;
    for(int64_t job=0;job<jobs;++job) {
        rc=check_job(p,job);if(rc!=RAD_OK) {*p.status=rc;return rc;}
    }
    for(int64_t job=0;job<jobs;++job) {
        float gate=0;evaluate(p,job,&gate);write_gate(p,job,gate);
        for(int64_t d=0;d<p.dim;++d)write_element(p,job,d,gate);
    }
    return RAD_OK;
}
} // namespace rad::ds41::engram_gate
