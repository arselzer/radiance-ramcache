Exact Core V7 reconstruction over the current Core V8 source

Apply these FOUR archived Engine paths to a COPY of radiance/, preserving all
other paths. The resulting 1535-file map must equal the untouched
research/glm53-main-verified-source-engine-v9-prototype/CORE_SOURCE_PINS.json
SHA753436bac31f041b8e051fb4e4797319d4eefa12df6d2bde0f8e28eee407b366.
The current V8 map is df4051bb85328badd3a130953721d31efff36b94133560bca29fcbb758a1ee0e.
Only core/engine.cpp, core/engine.h, core/engine_glm_main.cpp and
core/engine_glm_main.h differ. These copies are byte-exact frozen V7 sources.

The full public d04b562 checkpoint is separately reconstructible using the
base-d04b562-checkpoint manifest and its archived overwritten wrapper/docs,
while every historical raw package/evidence file remains unchanged.
After restoring V7, the existing six-file historical-core-v6-overlay restores
Core V6; the earlier two-file V5 overlay then restores Core V5. Always verify
the complete matching maps; never compile mixed private headers and archives.
