#pragma once
#include "rad_plugin.h"
namespace glm53_tiny_native_arch {
int declare(RadBuilder*,const RadModelMeta*,const RadBuildCtx*);
void step(RadCtx*,const RadBatch*);
}
