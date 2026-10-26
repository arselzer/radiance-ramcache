// Parent-only manual actual complete-source Engine; no CTest or auto-run.
#include "engine.h"
#include "engine_glm_main.h"
#include "device/device.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
using namespace rad;
int main(int argc,char** argv){
    try{
        const bool text=argc==8&&!std::strcmp(argv[1],"--one-token");
        const bool generate=argc>=9&&!std::strcmp(argv[1],"--generate");
        const bool formatted=argc>=9&&!std::strcmp(argv[1],"--prefill4-formatted");
        const bool prefill=formatted||(argc>=9&&!std::strcmp(argv[1],"--prefill4"));
        const bool transport=argc==7&&!std::strcmp(argv[1],"--transport");
        if(!text&&!transport&&!generate&&!prefill){std::cerr<<"usage: glm53_main_verified_engine_manual --transport MODEL EXPERIMENTAL_HOME COMPLETE_BUNDLE PRODUCER_PROOF TRUSTED_ID\n       ... --one-token MODEL EXPERIMENTAL_HOME COMPLETE_BUNDLE PRODUCER_PROOF TRUSTED_ID ONE_TOKEN_TEXT\n       ... --generate MODEL EXPERIMENTAL_HOME COMPLETE_BUNDLE PRODUCER_PROOF TRUSTED_ID OUTPUTS ONE_TOKEN_TEXT [--hot-mib N --host-payload-gib N --read-chunk-kib 64|1024]\n       ... --prefill4 MODEL EXPERIMENTAL_HOME COMPLETE_BUNDLE PRODUCER_PROOF TRUSTED_ID OUTPUTS TEXT [same flags]\n       ... --prefill4-formatted MODEL EXPERIMENTAL_HOME COMPLETE_BUNDLE PRODUCER_PROOF TRUSTED_ID OUTPUTS CALLER_RENDERED_TEXT [same flags]\n";return 2;}
        const std::string id=argv[6];if(id.empty()||id.size()>20||!std::all_of(id.begin(),id.end(),[](unsigned char c){return c>='0'&&c<='9';}))return 2;
        GlmMainEngineOptions options;options.checkpoint=argv[2];options.home=argv[3];options.bundle_root=argv[4];options.source_proof=argv[5];options.expected_bundle_id=std::stoull(id);
        uint32_t outputs=1;
        auto decimal=[](const char* raw,uint64_t* out){const std::string value=raw;if(value.empty()||value.size()>10||!std::all_of(value.begin(),value.end(),[](unsigned char c){return c>='0'&&c<='9';}))return false;*out=std::stoull(value);return true;};
        if(generate||prefill){options.prefill_max_tok=prefill?4:1;options.formatted_prompt=formatted;uint64_t count=0;if(!decimal(argv[7],&count)||count<(prefill?1u:2u)||count>glm_main_max_outputs)return 2;outputs=uint32_t(count);
            unsigned seen=0;for(int i=9;i<argc;i+=2){if(i+1>=argc)return 2;uint64_t value=0;if(!decimal(argv[i+1],&value))return 2;
                if(!std::strcmp(argv[i],"--hot-mib")){if((seen&1)||value>2048)return 2;seen|=1;options.hot_bytes_per_rank=value<<20;}
                else if(!std::strcmp(argv[i],"--host-payload-gib")){if((seen&2)||value<1||value>16)return 2;seen|=2;options.host_payload_limit=value<<30;}
                else if(!std::strcmp(argv[i],"--read-chunk-kib")){if((seen&4)||(value!=64&&value!=1024))return 2;seen|=4;options.read_chunk_bytes=uint32_t(value<<10);}
                else return 2;}}
        std::unique_ptr<Engine> engine;int status=GlmMainEngineHarness::create(options,&engine);
        if(status<0){std::cerr<<"request refused: "<<status<<'\n';return 1;}
        status=engine->start();int32_t token=-1;
        std::array<int32_t,glm_main_max_outputs> tokens{};
        if(status>=0&&text)status=GlmMainEngineHarness::one_token(*engine,argv[7],&token);
        if(status>=0&&(generate||prefill)){status=GlmMainEngineHarness::generate(*engine,argv[8],std::span<int32_t>(tokens.data(),outputs));if(status>=0)token=tokens[0];}
        GlmMainEngineSnapshot snapshot;GlmMainEngineHarness::snapshot(*engine,&snapshot);
        nlohmann::ordered_json report{{"schema","glm53-actual-verified-main-engine-v1"},{"passed",status==RAD_OK},
            {"status",status},{"phase",uint32_t(snapshot.phase)},{"failed_phase",uint32_t(snapshot.failed_phase)},
            {"source_admitted",snapshot.source_admitted},{"transport_ready",snapshot.transport_ready},
            {"text_complete",snapshot.text_complete},{"sampled_token",token},{"bundle_id",snapshot.bundle_id},
            {"native_bytes_verified",snapshot.source_bytes_verified},{"raw_mapped_bytes_compared",snapshot.mapped_bytes_compared},
            {"max_tok",options.prefill_max_tok},{"max_seqs",1},{"max_spec",0},{"tp_wire","exact"},{"fixed_residents",0},
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
        if(generate||prefill){report["schema"]=prefill?"glm53-actual-verified-main-prefill4-v1":"glm53-actual-verified-main-generate-v1";report["input_tokens"]=snapshot.input_tokens;
            if(prefill){report["formatted_prompt_parse_special"]=options.formatted_prompt;report["prompt_token_ids"]=nlohmann::ordered_json::array();
                for(size_t i=0;i<snapshot.input_tokens&&i<glm_main_max_prompt_tokens;++i)report["prompt_token_ids"].push_back(snapshot.prompt_token_ids[i]);}
            report["requested_outputs"]=outputs;report["generated_outputs"]=snapshot.generated_outputs;report["completed_steps"]=snapshot.completed_steps;
            report["generated_tokens"]=nlohmann::ordered_json::array();
            for(size_t i=0;i<snapshot.generated_outputs&&i<glm_main_max_outputs;++i)report["generated_tokens"].push_back(snapshot.generated_tokens[i]);
            report["startup_elapsed_ns"]=snapshot.startup_elapsed_ns;report["request_elapsed_ns"]=snapshot.request_elapsed_ns;
            report["sink_first_token_ns"]=snapshot.first_token_ns;report["sink_last_token_ns"]=snapshot.last_token_ns;
            report["manual_fixed_host_bytes"]=snapshot.manual_fixed_host_bytes;
            report["hot_bytes_per_rank"]=options.hot_bytes_per_rank;report["host_payload_limit"]=options.host_payload_limit;
            report["read_chunk_bytes"]=options.read_chunk_bytes;
            report["steps"]=nlohmann::ordered_json::array();
            for(size_t i=0;i<snapshot.completed_steps&&i<glm_main_max_steps;++i){const auto& w=snapshot.steps[i];
                report["steps"].push_back({{"epoch",w.epoch},{"step",w.step},{"phase",w.phase},{"rows",w.rows},{"n_ahead",w.n_ahead},
                    {"request",w.request},{"context_position",w.context_position},{"input_token",w.input_token},{"sampled_token",w.sampled_token},
                    {"completed_prepare_gates_per_rank",w.completed_prepare_gates},{"model_sampler_collect_complete",w.model_sampler_collect_complete},
                    {"begin_monotonic_ns",w.begin_ns},{"end_monotonic_ns",w.end_ns},{"elapsed_ns",w.end_ns-w.begin_ns}});
                if(prefill){auto& step=report["steps"].back();step["model_complete"]=w.model_complete;step["sample_expected"]=w.sample_expected;
                    step["sampler_collect_performed"]=w.sampler_collect_performed;step["input_tokens"]=w.input_tokens;}}
            report["provider_stats"]=nlohmann::ordered_json::array();
            for(const auto& p:snapshot.provider_stats)report["provider_stats"].push_back({{"acquires",p.acquires},{"refused",p.refused},
                {"hot_hits",p.hot_hits},{"cold_hits",p.cold_hits},{"misses",p.misses},{"duplicates",p.duplicates},
                {"payload_opens",p.payload_opens},{"payload_reads_records",p.payload_reads},{"payload_read_bytes",p.payload_read_bytes},
                {"fd_evictions",p.fd_evictions},{"stat_probes",p.stat_probes},{"evictions",p.evictions},{"admission_bypasses",p.admission_bypasses},
                {"hot_bytes",p.usage.hot_bytes},{"cold_bytes",p.usage.cold_bytes},{"peak_hot_bytes",p.usage.peak_hot_bytes},{"peak_cold_bytes",p.usage.peak_cold_bytes}});
            report["counter_scope"]="actual quiescent provider counters; payload_reads counts records, payload_read_bytes is buffered pread bytes, not physical NVMe traffic; no mover timing or checksum shortcut";}
        if(status>=0)GlmMainEngineHarness::retire(*engine);
        std::cout<<report.dump(2)<<'\n';return status==RAD_OK?0:1;
    }catch(const std::exception& e){std::cerr<<"manual actual Engine: "<<e.what()<<'\n';return 1;}
}
