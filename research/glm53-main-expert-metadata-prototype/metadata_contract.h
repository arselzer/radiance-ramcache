#pragma once
#include "main_metadata.h"
#include <cstdint>
namespace glm53::main_metadata {
struct ContractReceipt {uint64_t checks=0,refusals=0;};
// Parent-run API contract cases against its ACTUAL declared rank Program/Plan.
// Each mutation is one field and restored before the next call; no W snapshot.
int contract_checks(rad::Checkpoint&,rad::CheckpointSources&,rad::Program&,rad::Plan&,
                    ContractReceipt*,std::string* why=nullptr) noexcept;
}
