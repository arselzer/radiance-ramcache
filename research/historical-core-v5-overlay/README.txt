Exact historical CoreV5 source overlay

Original V5 pins are research/glm53-main-verified-source-engine-v5-prototype/CORE_SOURCE_PINS.json
SHA2560001f0a163dbc87198b899f9f9c782e6a9a0071c7a6a964fb2d2e10a68b8f19e.
Current radiance/ has the identical1535-file set and identical contents for1533
files. The two replacements archived here are exact V5 loader.h/.cpp. Overlaying
these2files onto a copy of radiance/ reconstructs ALL originalV5pins exactly.
This preserves source/old evidence scopes without duplicating upstream binary
test assets or pretending oldV5checks validate newCoreV6. No automatic script.
