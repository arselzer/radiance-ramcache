// Explicitly gated corrective derivative. External all-T Q/S lifetime is PERSIST.
#ifndef RAD_HAVE_HIP
#define RAD_HAVE_HIP 1
#endif
// Original data/oracle/operators/failure matrix preserved; never used by serving.
#include "adapter_fixture.h"
#include "ffn_oracle.h"
#include <cfenv>
#include <limits>
extern "C" int __real_rad_stream_sync(RadStream);
extern "C" int __wrap_rad_stream_sync(RadStream stream){const int result=__real_rad_stream_sync(stream);if(result<0)contain("actual negative stream drain (manual fail-stop wrapper)");return result;}
namespace io_fault {int remaining=-1;uint64_t failures=0;}
extern "C" ssize_t __real_pread(int,void*,size_t,off_t);
extern "C" ssize_t __wrap_pread(int fd,void* p,size_t n,off_t at){if(io_fault::remaining>=0&&io_fault::remaining--==0){io_fault::remaining=-1;++io_fault::failures;errno=EIO;return -1;}return __real_pread(fd,p,n,at);}
namespace wait_fault {RadStream stream=nullptr;int remaining=-1;uint64_t failures=0;}
extern "C" int __real_rad_event_wait(RadStream,RadEvent);
extern "C" int __wrap_rad_event_wait(RadStream stream,RadEvent event){if(stream==wait_fault::stream&&wait_fault::remaining>=0&&wait_fault::remaining--==0){wait_fault::remaining=-1;++wait_fault::failures;return RAD_E_IO;}return __real_rad_event_wait(stream,event);}
namespace {
size_t forwards=0,refused=0,exact_bf16=0,exact_fp8=0,exact_scales=0,hot_passes=0,resident_metadata_refusals=0;uint64_t max_router_weight_ulp=0;
size_t snapshot_slices=0,snapshot_bf16=0,snapshot_fp8=0,snapshot_scales=0,input_words=0;

std::array<std::vector<std::byte>,E> records(int layer){std::array<std::vector<std::byte>,E> out;
 for(int e=0;e<E;++e){auto& b=out[size_t(e)];b.assign(32768,std::byte{});for(int p=0;p<2;++p){const int rows=p?128:256;const size_t code=p?20480:0,scale=p?28672:16384;
  for(int n=0;n<rows;++n){for(int k=0;k<H;k+=2)b[code+size_t(n*H+k)/2]=std::byte(native_code(layer,e+p,n,k)|(native_code(layer,e+p,n,k+1)<<4));for(int g=0;g<H/32;++g)b[scale+size_t(n*4+g)]=std::byte(native_exponent(layer,e+p,n,g));}}}
 return out;
}
template<class X> std::vector<X> read(Fixture& f,rad_buf buf,size_t count){const auto raw=f.readback->read(f.ctx.buf_ptr(buf),count*sizeof(X));std::vector<X> out(count);std::memcpy(out.data(),raw.data(),raw.size());return out;}
void check_snapshots(Fixture& f,const ffn_reference::Result& expected){
 need(f.n_slices==size_t((T+R-1)/R)&&f.snapshot_used<=size_t(Stage::bytes),"every actual supplied/learned slice captured before transient reuse");
 std::array<int32_t,T*K> order{};for(int i=0;i<T*K;++i)order[size_t(i)]=i;
 std::stable_sort(order.begin(),order.end(),[&](int a,int b){return f.expected_ids[size_t(a)]<f.expected_ids[size_t(b)];});
 need(std::all_of(f.snapshots->host+f.snapshot_used,f.snapshots->host+Stage::bytes,[](uint8_t b){return b==0xA5;}),"unused snapshot bank tail changed");
 int64_t next=0;
 for(size_t si=0;si<f.n_slices;++si){const auto& s=f.slices[si];need(s.first==next&&s.rows==std::min<int64_t>(R,T-next),"exact2/2/1 snapshot geometry");next+=s.rows;
  const auto sorted=f.saved<int32_t>(s.sorted,size_t(s.rows*K));const auto gu=f.saved<uint16_t>(s.gu,size_t(s.rows*K*2*H));const auto q=f.saved<uint8_t>(s.q,size_t(s.rows*K*H));const auto scale=f.saved<float>(s.scale,size_t(s.rows*K));const auto dn=f.saved<uint16_t>(s.dn,size_t(s.rows*K*H));const auto gathered=f.saved<uint16_t>(s.gather,size_t(s.rows*H));
  std::array<bool,R*K> seen{};int last_expert=-1;
  for(int64_t row=0;row<s.rows*K;++row){const int local=sorted[size_t(row)];need(local>=0&&local<s.rows*K&&!seen[size_t(local)],"current slice sort is not a complete local permutation");seen[size_t(local)]=true;
   const int flat=int(s.first*K)+local;const auto expert=f.expected_ids[size_t(flat)];need(expert>=last_expert,"current slice expert run order");last_expert=expert;
   const auto found=std::find(order.begin(),order.end(),flat);need(found!=order.end(),"independent original job mapping");const size_t oracle=size_t(found-order.begin());
   const std::string at=" layer="+std::to_string(f.active_layer)+" slice="+std::to_string(si)+" global_token="+std::to_string(flat/K)+" slot="+std::to_string(flat%K)+" expert="+std::to_string(expert);
   for(int64_t n=0;n<2*H;++n){need(gu[size_t(row*2*H+n)]==expected.gu[oracle*2*H+size_t(n)],"all-slice independent GU BF16"+at+" col="+std::to_string(n));++snapshot_bf16;}
   for(int64_t n=0;n<H;++n){need(q[size_t(row*H+n)]==expected.q[oracle*H+size_t(n)],"all-slice independent FP8"+at+" col="+std::to_string(n));++snapshot_fp8;need(dn[size_t(row*H+n)]==expected.dn[oracle*H+size_t(n)],"all-slice independent DN BF16"+at+" col="+std::to_string(n));++snapshot_bf16;}
   need(std::bit_cast<uint32_t>(scale[size_t(row)])==std::bit_cast<uint32_t>(expected.scale[oracle]),"all-slice independent F32 scale"+at);++snapshot_scales;
  }
  for(int64_t row=0;row<s.rows;++row)for(int64_t n=0;n<H;++n){need(gathered[size_t(row*H+n)]==expected.gathered[size_t((s.first+row)*H+n)],"all-slice independent gathered BF16 global_token="+std::to_string(s.first+row)+" col="+std::to_string(n));++snapshot_bf16;}
  const auto iq=f.saved<uint8_t>(s.input_q,T*H);const auto is=f.saved<float>(s.input_s,T);
  for(size_t i=0;i<iq.size();++i){need(iq[i]==f.original_q[i],"full-pass external input FP8 changed after a slice");++input_words;}
  for(size_t i=0;i<is.size();++i){need(std::bit_cast<uint32_t>(is[i])==std::bit_cast<uint32_t>(f.original_scale[i]),"full-pass external input scale changed after a slice");++input_words;}
  ++snapshot_slices;
 }
 need(next==T,"all original tokens included in slice snapshots");
 const auto iq=read<uint8_t>(f,f.selected().w.h.q,T*H);const auto is=read<float>(f,f.selected().w.h.s,T);
 for(size_t i=0;i<iq.size();++i){need(iq[i]==f.original_q[i],"final full-pass input FP8 stability");++input_words;}
 for(size_t i=0;i<is.size();++i){need(std::bit_cast<uint32_t>(is[i])==std::bit_cast<uint32_t>(f.original_scale[i]),"final full-pass input scale stability");++input_words;}
}
void verify(Fixture& f){const auto expected=ffn_reference::evaluate(records(f.active_layer),f.expected_ids,f.original_q,f.original_scale,f.expected_weights);
 check_snapshots(f,expected);
 need(expected.positive_gates&&expected.negative_gates&&expected.clamped_gates&&expected.unclamped_pairs&&expected.positive_products&&expected.negative_products&&expected.distinct_codes>=16,"independent nonlinear/sign/quant fixture variety");
 for(size_t i=0;i<f.caller.size();++i){need(f.caller[i]==expected.gathered[i],"independent canonical complete FFN output index="+std::to_string(i)+" expected="+std::to_string(expected.gathered[i])+" observed="+std::to_string(f.caller[i]));++exact_bf16;}
 const auto ids=read<int32_t>(f,f.selected().w.ids,T*K);const auto weight=read<float>(f,f.selected().w.ew,T*K);
 for(size_t i=0;i<ids.size();++i){need(ids[i]==f.expected_ids[i],"current native routing IDs differ from independent routing");
  need(std::isfinite(weight[i])&&weight[i]>0,"native route weight domain");const uint32_t a=std::bit_cast<uint32_t>(weight[i]),b=std::bit_cast<uint32_t>(f.expected_weights[i]);const uint64_t delta=a>b?a-b:b-a;max_router_weight_ulp=std::max(max_router_weight_ulp,delta);
  // Predeclared scalar-vs-OCML exp/div allowance; final BF16 output remains EXACT.
  need(delta<=(f.learned?4u:0u),"router F32 probability weight exceeds declared4ULP scalar/OCML allowance");}
 if(f.learned){const auto logits=read<float>(f,f.selected().w.logits,T*E);for(int row=0;row<T;++row){const float x=rad_bf16_to_f32(f.original_input[size_t(row*H)]),z=rad_bf16_to_f32(f.original_input[size_t(row*H+1)]);const std::array<float,E> exact{4*x,3*x,3*z,z};for(int e=0;e<E;++e)need(std::bit_cast<uint32_t>(logits[size_t(row*E+e)])==std::bit_cast<uint32_t>(exact[size_t(e)]),"actual DEVICE F32 router projection");}}
 // Compare the last ragged slice's actual stage rows against global independent jobs.
 std::array<int32_t,T*K> global{};for(int i=0;i<T*K;++i)global[size_t(i)]=i;
 std::stable_sort(global.begin(),global.end(),[&](int a,int b){return f.expected_ids[size_t(a)]<f.expected_ids[size_t(b)];});
 const auto sorted=read<int32_t>(f,f.selected().w.sorted,K);const auto gu=read<uint16_t>(f,f.selected().w.egu,K*2*H);const auto dn=read<uint16_t>(f,f.selected().w.edn,K*H);const auto q=read<uint8_t>(f,f.selected().w.eff.q,K*H);const auto scales=read<float>(f,f.selected().w.eff.s,K);
 for(int row=0;row<K;++row){need(sorted[size_t(row)]>=0&&sorted[size_t(row)]<K,"ragged slice sorted local coordinate");const int flat=(T-1)*K+sorted[size_t(row)];const auto at=std::find(global.begin(),global.end(),flat);need(at!=global.end(),"independent job index");const size_t oracle=size_t(at-global.begin());
  for(int n=0;n<2*H;++n){need(gu[size_t(row*2*H+n)]==expected.gu[oracle*2*H+size_t(n)],"ragged GU BF16 stage");++exact_bf16;}
  for(int n=0;n<H;++n){need(dn[size_t(row*H+n)]==expected.dn[oracle*H+size_t(n)],"ragged DN BF16 stage");++exact_bf16;need(q[size_t(row*H+n)]==expected.q[oracle*H+size_t(n)],"ragged fused BF16-product/FP8 stage");++exact_fp8;}
  need(std::bit_cast<uint32_t>(scales[size_t(row)])==std::bit_cast<uint32_t>(expected.scale[oracle]),"ragged FP8 F32 scale stage");++exact_scales;
 }
 f.check_guards();need(f.owner->stats().held_leases==0&&f.source->adapter_usage().live_holders==0,"actual last reader retained source lease");++forwards;
}
void terminal(Fixture& f,int first){need(first<0&&f.ctx.terminal_status()==first,"strict DEVICE failure did not latch");const auto caller=f.caller;const auto version=f.caller_version;const uint64_t io=f.source->stats().payload_reads,source=f.owner->stats().source_batches,epoch=f.mover.residency()->epoch(),copies=f.ctx.expert_prepare_copies(),entries=f.step_entries;const auto waits=f.ctx.stream_waits();const auto totals=payload_totals();const std::string diagnostic=f.ctx.step_error();need(!diagnostic.empty(),"owned first diagnostic");
 const auto input=read<uint16_t>(f,f.selected().w.h.x,T*H);RadBatch b{};b.n_tok=T;b.n_seq=1;b.phase=RAD_PHASE_PREFILL;active=&f;const int direct=f.ctx.run_step(&b);active=nullptr;
 need(direct==first&&f.run()==first&&f.ctx.expert_prepare(nullptr)==first&&f.ctx.issue(f.selected().op_gu,nullptr,0,R)==first&&f.ctx.set_lane(1)==first,"direct Core terminal guards");
 const auto h=f.plan.units[size_t(residents(f).front())].weights.front();need(!f.ctx.weight_ptr(h),"terminal weight query enqueued readiness");
 need(f.caller==caller&&f.caller_version==version&&f.source->stats().payload_reads==io&&f.owner->stats().source_batches==source&&f.mover.residency()->epoch()==epoch&&f.ctx.expert_prepare_copies()==copies&&f.ctx.stream_waits()==waits&&f.step_entries==entries&&payload_totals()==totals&&std::string(f.ctx.step_error())==diagnostic&&read<uint16_t>(f,f.selected().w.h.x,T*H)==input,"terminal retry performed new model work/publication");
 need(f.owner->stats().held_leases==0&&f.source->adapter_usage().live_holders==0,"error retirement retained healthy-stream leases");f.check_guards();++refused;
 auto* program=&f.builder->program();f.ctx.shutdown();const auto after=payload_totals();CtxDesc d{};d.program=program;d.stream=f.compute.value;d.residency=f.mover.residency();d.arch_step=step;d.record_passes=false;d.expert_prepare_owner=f.owner.get();need(f.ctx.init(d)==first&&payload_totals()==after&&std::string(f.ctx.step_error())==diagnostic,"same Ctx reinit bypassed terminal DEVICE state");
}
void positive(const std::string& home,const std::string& plugin,bool learned){++cases;Fixture f(home,plugin);f.learned=learned;f.mode=3;
 need(f.producer_retained==0&&f.initial.capacity()==0&&f.specs.capacity()==0&&f.source_catalog.tensors.empty(),"old producer retained at forward");
 std::vector<std::pair<rad_weight,WeightSlot>> fixed;
 for(const auto u:residents(f))for(const auto h:f.plan.units[size_t(u)].weights)fixed.push_back({h,*f.mover.residency()->lookup(h)});
 const auto native=f.inventory->footprint();need(native.native_device_bytes==196608&&native.landing_bytes==196608&&native.owned_units==24&&native.regions==4&&native.consumer_slots==3,"actual P3-sized native allocation geometry");
 uint64_t ctx=0;for(size_t i=0;i<f.before_ctx.size();++i)ctx+=uint64_t(f.after_ctx[i].second-f.before_ctx[i].second);
 const auto usage=f.source->process_usage();const auto component=f.device_allocator.physical+f.host_allocator.physical+usage.metadata_reserved+usage.cache_reserved+native.metadata_reservation+65536+ctx+Fixture::strict_extra;
 need(component<Fixture::admitted&&f.before_mover==f.after_mover,"closed DEVICE/Pools/source component admission or old hidden expert allocation");
 for(const auto [layer,mode]:std::array<std::pair<int,int>,5>{{{0,0},{0,0},{7,2},{7,2},{3,3}}}){f.active_layer=layer;f.mode=mode;const auto before=f.source->stats();good(f.run(),"actual current-route Native AQL FFN");verify(f);const auto after=f.source->stats();if((layer==0&&f.caller_version==2)||(layer==7&&f.caller_version==4)){need(after.payload_reads==before.payload_reads&&after.payload_opens==before.payload_opens,"same immutable source generation HOT reread");++hot_passes;}}
 need(f.owner->stats().peak_held_leases==3&&f.source->stats().hot_hits,"whole-union/strong lease/global HOT evidence");
 for(const auto& [h,original]:fixed){const auto* now=f.mover.residency()->lookup(h);need(now&&now->ptr==original.ptr&&now->ready==original.ready&&now->generation==original.generation,"static expert0 Residency changed");}
 for(const auto u:residents(f)){const auto& unit=f.plan.units[size_t(u)];const auto raw=f.readback->read(f.mover.unit_base(u),size_t(unit.bytes));const auto expected=records(unit.layer);need(std::equal(raw.begin(),raw.end(),expected[0].begin()),"static expert0 DEVICE source bytes changed");}
 std::cout<<"COMPONENT bytes="<<component<<" admitted="<<Fixture::admitted<<" declared="<<Fixture::declared_upper<<" native="<<native.native_device_bytes<<" landing="<<native.landing_bytes<<" metadata="<<native.metadata_reservation<<" pool-physical="<<f.device_allocator.physical+f.host_allocator.physical<<" mapped-helper-reservation="<<Fixture::strict_extra<<" snapshot-bank-mapped="<<Stage::allocation_bytes<<" snapshot-payload-peak="<<f.snapshot_used<<" actual-arena="<<f.builder->program().arena_bytes<<" router="<<(learned?"actual DEVICE learned-form synthetic weights":"supplied legal F32 routes")<<'\n';
}
void failures(const std::string& home,const std::string& plugin){++cases;for(int kind=0;kind<5;++kind){Fixture f(home,plugin);f.mode=3;good(f.run(),"publication before fatal DEVICE fault");verify(f);const auto caller=f.caller;const auto published=f.owner->stats().published_units;f.active_layer=7;
 const uint64_t pread_before=io_fault::failures,copy_before=copy_fault::failed,event_before=event_fault::failed,wait_before=wait_fault::failures;
 if(kind==0)io_fault::remaining=1;
 if(kind==1){copy_fault::stream=f.mover.transfer_stream();copy_fault::remaining=0;}
 if(kind==2){wait_fault::stream=f.compute.value;wait_fault::remaining=0;}
 if(kind==3)f.fail_dn=true;
 if(kind==4){event_fault::stream=f.mover.transfer_stream();event_fault::remaining=0;}
 const int error=f.run();io_fault::remaining=-1;copy_fault::stream=nullptr;copy_fault::remaining=-1;wait_fault::stream=nullptr;wait_fault::remaining=-1;event_fault::stream=nullptr;event_fault::remaining=-1;
 const std::string context="fault kind="+std::to_string(kind)+" status="+std::to_string(error)+
     " pread_delta="+std::to_string(io_fault::failures-pread_before)+" copy_delta="+std::to_string(copy_fault::failed-copy_before)+
     " event_delta="+std::to_string(event_fault::failed-event_before)+" wait_delta="+std::to_string(wait_fault::failures-wait_before);
 need(io_fault::failures==pread_before+uint64_t(kind==0)&&copy_fault::failed==copy_before+uint64_t(kind==1)&&
      event_fault::failed==event_before+uint64_t(kind==4)&&wait_fault::failures==wait_before+uint64_t(kind==2),
      "intended semantic wrapper did not trigger exactly once: "+context);
 if(kind!=3){need(error==RAD_E_IO,"injected semantic fault returned an unrelated error: "+context);}
 else need(error<0&&std::string(f.ctx.step_error()).find("invalid.late.DN")!=std::string::npos,"DN refusal was unrelated to the targeted bad operand: "+context);
 need(error<0&&f.caller==caller&&f.caller_version==1,"fatal pass committed caller bytes/version");
 if(kind==0||kind==1||kind==4){need(f.owner->stats().published_units==published,"source/copy/record refusal published incomplete Native union");}
 terminal(f,error);
 }}
void profile_refusals(const std::string& home,const std::string& plugin){++cases;Fixture f(home,plugin);const auto bindings=f.bound(*f.source);const auto fixed=residents(f);const auto base=f.inventory->limits();const auto io=f.source->stats().payload_reads;const auto alloc=payload_totals();
 for(int kind=0;kind<4;++kind){auto limits=base;if(kind==0)limits.backend_profile=NativeBackendProfile(99);if(kind==1)limits.max_rows=3;if(kind==2)limits.top_k=3;if(kind==3)limits.max_host_bytes=128u<<20;std::shared_ptr<const NativeInventory> out;need(NativeInventory::bind(f.builder->program(),f.metadata_plan,f.source->catalog(),bindings,fixed,limits,&out)<0&&!out&&f.source->stats().payload_reads==io&&payload_totals()==alloc,"unsupported profile changed payload admission/source");}
 // All eight static expert0 units and each of their four members remain canonical.
 // Restore the live Program before any assertion/cleanup, even if admission is wrong.
 for(const auto u:fixed)for(const auto h:f.metadata_plan.units[size_t(u)].weights){
     auto& w=f.builder->program().weights[h];const auto encoding=w.enc;
     w.enc.transform[0]='x';w.enc.transform[1]=0;
     std::shared_ptr<const NativeInventory> out;const auto before=f.source->stats();
     const int status=NativeInventory::bind(f.builder->program(),f.metadata_plan,f.source->catalog(),bindings,fixed,base,&out);
     w.enc=encoding;
     const auto after=f.source->stats();
     need(status==RAD_E_DTYPE&&!out&&after.acquires==before.acquires&&after.payload_reads==before.payload_reads&&
          after.payload_opens==before.payload_opens&&payload_totals()==alloc,
          "noncanonical resident member passed pure metadata admission or performed I/O/payload/output mutation");
     ++resident_metadata_refusals;
 }
 Config cfg=f.pool_config;PlannerInput pi{&f.builder->program(),&cfg,0,1,nullptr,0,false,true};pi.native_inventory=f.inventory;Plan rejected;
 need(Planner(pi).plan(&rejected)==RAD_E_UNSUPPORTED&&f.source->stats().payload_reads==io&&payload_totals()==alloc,"AQL inventory admitted a falsely labelled HOST planner profile");
 need(f.mover.native_record_guards_host()==RAD_E_UNSUPPORTED,"DEVICE enabled legacy HOST guard helper");
}
}
int main(int argc,char** argv){try{const bool host_kargs=argc==7&&!std::strcmp(argv[6],"--host-kargs");need((argc==6||host_kargs)&&!std::strcmp(argv[1],"--run-gpu")&&!std::strcmp(argv[2],"--radiance-home")&&!std::strcmp(argv[4],"--plugin"),"usage: CHECKER --run-gpu --radiance-home MATCHING_HOME --plugin MODULE [--host-kargs]");if(host_kargs)good(aql_force_host_kargs(),"force actual host kernargs before backend initialization");need(std::numeric_limits<float>::is_iec559&&std::fegetround()==FE_TONEAREST,"IEEE F32 RNE required");need(rad_dev_count()==1,"exactly one externally selected visible GPU required");good(rad_dev_set(0),"select one visible card");RadDeviceProps props{};good(rad_dev_props(0,&props),"device metadata");need(!props.is_host_backend&&!std::strcmp(props.arch,"gfx1201")&&!std::strcmp(device_backend_name(),"aql"),"actual gfx1201 AQL only");need(rad_core_model_failure_policy_version()==RAD_MODEL_FAILURE_POLICY_VERSION,"matching PermanentV1 Core");positive(argv[3],argv[5],false);positive(argv[3],argv[5],true);failures(argv[3],argv[5]);profile_refusals(argv[3],argv[5]);need(cases==4&&forwards==15&&refused==5&&hot_passes==4&&resident_metadata_refusals==32,"honest case/forward/refusal counters");need(snapshot_slices==45&&snapshot_bf16==67200&&snapshot_fp8==19200&&snapshot_scales==150&&input_words==38700,"all original slice/input stability categories");need(io_fault::failures==1&&copy_fault::failed==1&&event_fault::failed==1&&wait_fault::failures==1,"honest injected semantic fault trigger totals");std::cout<<"PASS "<<cases<<" actual Native DEVICE-owner cases, "<<checks<<" checks; forwards="<<forwards<<" refused="<<refused<<" exact_BF16="<<exact_bf16<<" exact_FP8="<<exact_fp8<<" exact_scale="<<exact_scales<<" resident_metadata_refusals="<<resident_metadata_refusals<<" semantic_fault_triggers="<<io_fault::failures<<','<<copy_fault::failed<<','<<event_fault::failed<<','<<wait_fault::failures<<" router_max_ULP="<<max_router_weight_ulp<<" snapshot_slices="<<snapshot_slices<<" snapshot_BF16="<<snapshot_bf16<<" snapshot_FP8="<<snapshot_fp8<<" snapshot_scales="<<snapshot_scales<<" input_words="<<input_words<<" kernargs="<<(host_kargs?"host":"default")<<"; P3-sized TP1 AQL synthetic, no Engine/checkpoint/TP/speculation/failed-drain recovery claim\n";return 0;}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
