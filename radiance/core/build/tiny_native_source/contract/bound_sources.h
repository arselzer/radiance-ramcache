#pragma once
#include "profile.h"
#include "build/startup.h"
namespace glm53::engine_core_contract {
// Actual unchanged Core CheckpointSources, with an inseparable retained owner.
// Member destruction order retires the cache before releasing its Checkpoint.
// No raw borrowed cache can be supplied to the bound quote interface.
class BoundSources {
public:
    explicit BoundSources(const ColdSourceSeal& seal):owner_(seal.owner_),actual_(owner_->checkpoint()){}
    BoundSources(const BoundSources&)=delete;
    rad::WeightSourceFn fn(){return actual_.fn();}
    const rad::CkptWeight* find(const std::string& name)const{return actual_.find(name);}
    std::string missing(const std::string& name)const{return actual_.missing(name);}
    std::vector<std::string> refused()const{return actual_.refused();}
    const SourceCheckpoint* owner_identity()const noexcept{return owner_.get();}
private:
    std::shared_ptr<const SourceCheckpoint> owner_;
    rad::CheckpointSources actual_;
};
} // namespace glm53::engine_core_contract
