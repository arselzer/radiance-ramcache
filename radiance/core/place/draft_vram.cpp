/* draft_vram.cpp -- the hot-set file of RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM (draft_vram.h). */
#include "draft_vram.h"

#include <nlohmann/json.hpp>

#include <cerrno>
#include <cstdio>

namespace rad {

int draft_hotset_parse(const std::string& text, DraftHotset* out, std::string* why) {
    auto no = [&](std::string s) { if (why) *why = std::move(s); return (int)RAD_E_INVAL; };
    if (!out) return RAD_E_INVAL;
    nlohmann::json j;
    try { j = nlohmann::json::parse(text); }
    catch (const nlohmann::json::exception& e) { return no(std::string("not JSON: ") + e.what()); }
    if (!j.is_object() || !j.contains("format") || !j["format"].is_string() ||
        j["format"].get<std::string>() != kDraftHotsetFormat)
        return no(std::string("not a ") + kDraftHotsetFormat + " file");
    auto int_of = [&](const char* k, int64_t* v) {
        if (!j.contains(k) || !j[k].is_number_integer()) return false;
        *v = j[k].get<int64_t>();
        return true;
    };
    int64_t ranks = 0, experts = 0;
    if (!int_of("ranks", &ranks) || ranks != kDraftRanks)
        return no("\"ranks\" must be " + std::to_string(kDraftRanks) + " (the bank is served at EP4)");
    if (!int_of("experts", &experts) || experts != kDraftExperts)
        return no("\"experts\" must be " + std::to_string(kDraftExperts) + " (a DSpark stage's routed experts)");
    if (!j.contains("layers") || !j["layers"].is_array() || j["layers"].size() != (size_t)kDraftStages)
        return no("\"layers\" must name the three stages 40, 41, 42");
    for (int s = 0; s < kDraftStages; ++s) {
        const auto& l = j["layers"][(size_t)s];
        if (!l.is_number_integer() || l.get<int64_t>() != kDraftFirstLayer + s)
            return no("\"layers\" must name the three stages 40, 41, 42");
    }
    if (!j.contains("hot") || !j["hot"].is_array() || j["hot"].size() != (size_t)kDraftStages)
        return no("\"hot\" must hold one entry a stage");
    DraftHotset hs;
    hs.ranks = (int32_t)ranks;
    hs.experts = (int32_t)experts;
    hs.hot.assign((size_t)kDraftStages, {});
    for (int s = 0; s < kDraftStages; ++s) {
        const auto& st = j["hot"][(size_t)s];
        if (!st.is_array() || st.size() != (size_t)ranks)
            return no("\"hot\" stage " + std::to_string(kDraftFirstLayer + s) + " must hold one list a rank");
        hs.hot[(size_t)s].assign((size_t)ranks, {});
        for (int r = 0; r < ranks; ++r) {
            const auto& l = st[(size_t)r];
            if (!l.is_array() || l.empty() || l.size() > (size_t)kDraftRankExperts)
                return no("stage " + std::to_string(kDraftFirstLayer + s) + ", rank " + std::to_string(r) +
                          ": a list of 1..32 experts");
            std::vector<uint8_t> seen((size_t)experts, 0);
            for (const auto& v : l) {
                if (!v.is_number_integer()) return no("an expert id that is not an integer");
                const int64_t g = v.get<int64_t>();
                if (g < 0 || g >= experts)
                    return no("expert id " + std::to_string(g) + " out of range");
                if (seen[(size_t)g])
                    return no("stage " + std::to_string(kDraftFirstLayer + s) + ", rank " +
                              std::to_string(r) + ": expert " + std::to_string(g) + " listed twice");
                seen[(size_t)g] = 1;
                hs.hot[(size_t)s][(size_t)r].push_back((int32_t)g);
            }
        }
    }
    *out = std::move(hs);
    return RAD_OK;
}

int draft_hotset_load(const std::string& path, DraftHotset* out, std::string* why) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) {
        if (why) *why = "cannot read " + path + ": " + std::strerror(errno);
        return RAD_E_IO;
    }
    std::string text;
    char buf[1 << 14];
    while (const size_t n = std::fread(buf, 1, sizeof buf, f)) text.append(buf, n);
    const bool bad = std::ferror(f) != 0;
    std::fclose(f);
    if (bad) {
        if (why) *why = "reading " + path + " failed";
        return RAD_E_IO;
    }
    const int st = draft_hotset_parse(text, out, why);
    if (st == RAD_OK) out->path = path;
    return st;
}

}  /* namespace rad */
