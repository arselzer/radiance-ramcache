// Parent-only manual actual complete-source Engine; no CTest or auto-run.
#include "engine.h"
#include "engine_glm_main.h"
#include "device/device.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>
using namespace rad;
int main(int argc,char** argv){
    try{
        const bool text=argc==8&&!std::strcmp(argv[1],"--one-token");
        const bool transport=argc==7&&!std::strcmp(argv[1],"--transport");
        if(!text&&!transport){std::cerr<<"usage: glm53_main_verified_engine_manual --transport MODEL EXPERIMENTAL_HOME COMPLETE_BUNDLE PRODUCER_PROOF TRUSTED_ID\n       ... --one-token MODEL EXPERIMENTAL_HOME COMPLETE_BUNDLE PRODUCER_PROOF TRUSTED_ID ONE_TOKEN_TEXT\n";return 2;}
        const std::string id=argv[6];if(id.empty()||id.size()>20||!std::all_of(id.begin(),id.end(),[](unsigned char c){return c>='0'&&c<='9';}))return 2;
        GlmMainEngineOptions options;options.checkpoint=argv[2];options.home=argv[3];options.bundle_root=argv[4];options.source_proof=argv[5];options.expected_bundle_id=std::stoull(id);
        std::unique_ptr<Engine> engine;int status=GlmMainEngineHarness::create(options,&engine);
        if(status<0){std::cerr<<"request refused: "<<status<<'\n';return 1;}
        status=engine->start();int32_t token=-1;
        if(status>=0&&text)status=GlmMainEngineHarness::one_token(*engine,argv[7],&token);
        GlmMainEngineSnapshot snapshot;GlmMainEngineHarness::snapshot(*engine,&snapshot);
        nlohmann::ordered_json report{{"schema","glm53-actual-verified-main-engine-v1"},{"passed",status==RAD_OK},
            {"status",status},{"phase",uint32_t(snapshot.phase)},{"failed_phase",uint32_t(snapshot.failed_phase)},
            {"source_admitted",snapshot.source_admitted},{"transport_ready",snapshot.transport_ready},
            {"text_complete",snapshot.text_complete},{"sampled_token",token},{"bundle_id",snapshot.bundle_id},
            {"native_bytes_verified",snapshot.source_bytes_verified},{"raw_mapped_bytes_compared",snapshot.mapped_bytes_compared},
            {"max_tok",1},{"max_seqs",1},{"max_spec",0},{"tp_wire","exact"},{"fixed_residents",0},
            {"model_vocab_rows",snapshot.vocabulary.model_rows},{"valid_token_count",snapshot.vocabulary.valid_tokens},
            {"padded_vocab_rows",snapshot.vocabulary.padded_rows},
            {"actual_quantum",snapshot.actual_quantum},{"actual_generic_minimum",snapshot.actual_generic_minimum},
            {"actual_state_chunk",snapshot.actual_state_chunk},{"actual_attention_block",snapshot.actual_attention_block},
            {"metadata_requested_envelope",snapshot.metadata_reserved},{"host_payload_requested_envelope",snapshot.host_payload_reserved},
            {"thread_stacks",snapshot.thread_stack_bytes},{"backend_stream_requested",snapshot.backend_stream_requested},
            {"device_requested",snapshot.device_requested},{"weights_requested",snapshot.weights_requested},
            {"kv_requested",snapshot.kv_requested},{"arena_requested",snapshot.arena_requested},
            {"batch_pinned_requested",snapshot.batch_pinned_requested},{"batch_device_requested",snapshot.batch_device_requested},{"batch_events",snapshot.batch_events},{"event_requested",snapshot.event_requested},{"context_requested",snapshot.context_requested},{"owner_requested",snapshot.owner_requested},
            {"mover_requested",snapshot.mover_requested},{"inventory_metadata",snapshot.inventory_metadata},
            {"ordinary_jobs",snapshot.loader.completed_jobs},{"ordinary_upload_bytes",snapshot.loader.upload_bytes},
            {"ordinary_successful_drains",snapshot.loader.successful_drains},{"omitted_expert_views",snapshot.loader.quote.omitted_views},
            {"loader_input",snapshot.loader.quote.input_bytes},{"loader_output",snapshot.loader.quote.output_bytes},
            {"loader_column",snapshot.loader.quote.column_bytes},{"loader_metadata",snapshot.loader.quote.metadata_requested},
            {"memory_scope","requested owners and conservative named envelopes; excludes mapped checkpoint RSS/OS page cache, vendor/driver/stdlib allocator RSS"},
            {"performance_or_quality_claim",false},{"production_deployed",false}};
        if(status>=0)GlmMainEngineHarness::retire(*engine);
        std::cout<<report.dump(2)<<'\n';return status==RAD_OK?0:1;
    }catch(const std::exception& e){std::cerr<<"manual actual Engine: "<<e.what()<<'\n';return 1;}
}
