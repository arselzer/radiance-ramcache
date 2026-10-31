#!/usr/bin/env python3
"""Prepared source-only host cases; integration owner runs under build.lock/nice19."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("dspark_cost_extract", Path(__file__).with_name("dspark_cost_extract.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def events(k):
    rows = []
    anchors = [40, 41, 42+k, 43+2*k]
    for e, anchor in enumerate(anchors, 1):
        for depth in (1, 2):
            p = dict(v=1, event="proposal", seq=1, epoch=e, anchor_pos=anchor, pos=anchor+depth,
                     depth=depth, token=10+2*e+depth, predecessor=9+2*e+depth, candidate_count=16,
                     candidate_p=.8, score=4., margin=2., margin_unbounded=False, signal_valid=True,
                     status="pending", verified=None, reached=None, accepted=None, cancelled=False)
            verified = e != 4 and depth <= k
            o = dict(p, event="outcome", verified=verified, reached=verified,
                     accepted=True if verified else None,
                     status="accepted" if verified else "request_closed" if e == 4 else "verify_cap")
            rows.extend((p, o))
    return rows


def log(k, ns=None, capture=False):
    ns = ns or (k+1)*1_000_000
    lines = [mod.EARLY + f"on k={k} (RADIANCE_EXPERIMENTAL_DS41_DSPARK_EARLY_VERIFY); query_rows=2 "
             "unchanged at every action, observer=1, wait=step_end before issue_ahead, "
             f"logging={'capture' if capture else 'quiet'}; fixed measurement, no probability policy"]
    if capture:
        lines.append("DSpark read-only proposal capture: on (source fixture)")
        lines += [mod.prob.MARKER + json.dumps(e) for e in events(k)]
        for epoch, steady in ((2, True), (3, False)):
            cycle = dict(seq=1, epoch=epoch, next_epoch=epoch+1, verify_step=epoch,
                         k=k, cycle_ns=ns, wait_ns=1000, steady=steady, live=steady,
                         published=steady, accepted=k, proposal_invalid_depths=0, next_invalid_depths=0)
            lines.append(mod.CYCLE + json.dumps(cycle))
    t = dict(queries=4, query_rows=8, observers=4, waits=4, wait_ns=4000,
             transition=1, no_query=0, end_cancel=1, unpublished=0,
             invalid_queries=0, invalid_cycles=0, unclosed_epoch=0)
    lines.append(mod.TOTALS + " ".join(f"{key}={t[key]}" for key in mod.TOTAL_KEYS))
    for action in range(3):
        n, cost, wait, acc, emitted = (1, ns, 1000, k, 1+k) if action == k else (0, 0, 0, 0, 0)
        lines.append(mod.COST + f"k={action} n={n} cycle_ns={cost} wait_ns={wait} accepted={acc} emitted={emitted} "
                     f"mean_ms={cost*1e-6:.6f}; consecutive same-sequence/epoch-bound pure-decode cycles only; "
                     f"includes query/observer/wait and logging={'capture' if capture else mod.QUIET}")
    return lines


def fixture(base):
    frozen = dict(schema="ds41-proposal-metadata-v1", requests=[
        dict(run="original", seq=1, family="code", prompt_key="code-01", split_group="p1", split="train"),
        dict(run="original", seq=2, family="code", prompt_key="code-02", split_group="p2", split="heldout"),
        dict(run="original", seq=3, family="code", prompt_key="code-03", split_group="p3", split="train")])
    (base/"frozen.json").write_text(json.dumps(frozen))
    profile = dict(id="test-profile", hardware="explicit hardware/cache description",
                   engine_commit="a"*40, common_config_sha256="b"*64,
                   libraries={"libr4d.so": "c"*64, "deepseek41.so": "d"*64})
    (base/"code-hotset.json").write_text("fixture hotset bytes\n")
    family_profile = dict(id="code-profile", config_sha256="e"*64,
                          hotset=dict(path="code-hotset.json", sha256=mod.hashed(base/"code-hotset.json")))
    manifest = dict(schema="ds41-matched-cost-manifest-v1", measurement_profile="fixture only",
                    same_drafter_observer_wait=True, authority_assertion="operator attests matching deployment and config",
                    profile=profile, family_profiles=dict(code=family_profile),
                    frozen_split=dict(metadata="frozen.json", sha256=mod.hashed(base/"frozen.json"),
                                                       seed="ds41-prob-v1", heldout_fraction=.25), runs=[])
    for k in range(3):
        (base/f"k{k}.log").write_text("\n".join(log(k))+"\n")
        result = dict(concurrency=1, requests=[dict(name="code-01", family="code", text="target reply", error=None)],
                      stats=dict(requests=[dict(id=1)]))
        (base/f"k{k}.json").write_text(json.dumps(result))
        manifest["runs"].append(dict(run=f"code{k}", family="code", k=k, scope="cost_train", logging="quiet",
                                     log_level="info", log=f"k{k}.log", results=f"k{k}.json", arm=f"all-Ds-Db2-Dme{k}",
                                     profile_id=profile["id"], engine_commit=profile["engine_commit"],
                                     common_config_sha256=profile["common_config_sha256"],
                                     family_profile_id=family_profile["id"], config_sha256=family_profile["config_sha256"],
                                     hotset=dict(family_profile["hotset"]), libraries=dict(profile["libraries"]),
                                     provenance_assertion="operator checked source, binaries, settings and original hot set",
                                     request_mapping_assertion="fresh server seq1, no hidden generation or warmups",
                                     requests=[dict(seq=1, frozen_run="original", frozen_seq=1)]))
    return manifest


def write_manifest(base, manifest):
    path = base/"manifest.json"
    path.write_text(json.dumps(manifest))
    return path


class CostExtractionTest(unittest.TestCase):
    def test_quiet_costs_are_weighted_integer_means_and_match_existing_schema(self):
        with tempfile.TemporaryDirectory(prefix="ds41-cost-test-") as tmp:
            base = Path(tmp); manifest = fixture(base)
            run = copy.deepcopy(manifest["runs"][1]); run.update(run="code1repeat", log="repeat.log")
            (base/"repeat.log").write_text("independent repeat\n"+"\n".join(log(1, ns=4_000_000))+"\n")
            manifest["runs"].append(run)
            path = write_manifest(base, manifest)
            result = mod.extract(path)
            self.assertEqual([result["families"]["code"][f"c{k}_ms"] for k in range(3)], [1., 3., 3.])
            self.assertFalse(result["extraction"]["fit_performed"])
            out = base/"costs.json"; out.write_text(json.dumps(result))
            self.assertEqual(mod.prob.load_costs(out)["schema"], "ds41-proposal-costs-v1")
            self.assertEqual(mod.main(["--manifest", str(path), "--out", str(out)]), 0)

    def test_wrong_scopes_counts_means_yields_and_exclusions_are_refused(self):
        good = "\n".join(log(1))
        bad = [good.replace("on k=1", "on k=0"), good+"\n"+log(1)[0],
               good.replace("query_rows=8", "query_rows=9"), good.replace("observers=4", "observers=3"),
               good.replace("waits=4", "waits=0"), good.replace("invalid_cycles=0", "invalid_cycles=1"),
               good.replace("invalid_queries=0", "invalid_queries=5"), good.replace("transition=1", "transition=4"),
               good.replace("mean_ms=2.000000", "mean_ms=2.100000"), good.replace("emitted=2", "emitted=3"),
               good+"\nexperimental draft verify cap: k=1", good+"\nDSpark matched early unknown: marker",
               good+"\ndrafter 'dspark': sequence 1 proposes 7 8", good+"\nDSpark proposal capture: {}",
               good.replace("invalid_queries=0 invalid_cycles=0 ", "")]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(mod.Invalid):
                mod.parse_log(text.splitlines(), 1, "quiet")
        self.assertEqual(mod.parse_log(log(0), 0, "quiet")["costs"][0]["accepted"], 0)

    def test_manifest_provenance_training_split_and_receipt_authority_are_required(self):
        with tempfile.TemporaryDirectory(prefix="ds41-cost-test-") as tmp:
            base = Path(tmp); good = fixture(base)
            for field, value in (("libraries", {}), ("config_sha256", "f"*64), ("logging", "capture"),
                                 ("log_level", "debug"), ("profile_id", "foreign"), ("arm", "all-Ds0-Db2-Dme0"),
                                 ("request_mapping_assertion", ""), ("provenance_assertion", "")):
                bad = copy.deepcopy(good); bad["runs"][0][field] = value
                with self.subTest(field=field), self.assertRaises(mod.Invalid):
                    mod.extract(write_manifest(base, bad))
            bad = copy.deepcopy(good); bad["runs"][0]["requests"][0]["frozen_seq"] = 2
            with self.assertRaises(mod.Invalid): mod.extract(write_manifest(base, bad))
            bad = copy.deepcopy(good); bad["frozen_split"]["sha256"] = "0"*64
            with self.assertRaises(mod.Invalid): mod.extract(write_manifest(base, bad))
            bad = copy.deepcopy(good); bad["runs"] = bad["runs"][:2]
            with self.assertRaises(mod.Invalid): mod.extract(write_manifest(base, bad))
            (base/"k0.json").write_text(json.dumps(dict(concurrency=1, requests=[dict(name="other", family="code", text="reply")])))
            with self.assertRaises(mod.Invalid): mod.extract(write_manifest(base, good))

    def test_nonmonotone_means_are_never_sorted_or_clamped(self):
        with tempfile.TemporaryDirectory(prefix="ds41-cost-test-") as tmp:
            base = Path(tmp); manifest = fixture(base)
            (base/"k1.log").write_text("\n".join(log(1, ns=500_000)))
            with self.assertRaises(mod.Invalid): mod.extract(write_manifest(base, manifest))

    def test_family_hotsets_stay_distinct_and_only_within_family_configs_match(self):
        with tempfile.TemporaryDirectory(prefix="ds41-cost-test-") as tmp:
            base = Path(tmp); manifest = fixture(base)
            metadata = json.loads((base/"frozen.json").read_text())
            metadata["requests"].append(dict(run="original", seq=4, family="chat", prompt_key="chat-01", split_group="chat1", split="train"))
            (base/"frozen.json").write_text(json.dumps(metadata))
            manifest["frozen_split"]["sha256"] = mod.hashed(base/"frozen.json")
            (base/"chat-hotset.json").write_text("different original chat hotset\n")
            fp = dict(id="chat-profile", config_sha256="f"*64,
                      hotset=dict(path="chat-hotset.json", sha256=mod.hashed(base/"chat-hotset.json")))
            manifest["family_profiles"]["chat"] = fp
            for k in range(3):
                run = copy.deepcopy(manifest["runs"][k])
                run.update(run=f"chat{k}", family="chat", log=f"chat{k}.log", results=f"chat{k}.json",
                           family_profile_id=fp["id"], config_sha256=fp["config_sha256"], hotset=fp["hotset"],
                           requests=[dict(seq=1, frozen_run="original", frozen_seq=4)])
                (base/f"chat{k}.log").write_text("chat run\n"+"\n".join(log(k)))
                (base/f"chat{k}.json").write_text(json.dumps(dict(concurrency=1, requests=[dict(name="chat-01", family="chat", text="reply")])))
                manifest["runs"].append(run)
            result = mod.extract(write_manifest(base, manifest))
            self.assertEqual(set(result["families"]), {"code", "chat"})
            self.assertNotEqual(result["families"]["code"]["profile"]["hotset"], result["families"]["chat"]["profile"]["hotset"])
            bad = copy.deepcopy(manifest); bad["runs"][-1]["hotset"] = bad["family_profiles"]["code"]["hotset"]
            with self.assertRaises(mod.Invalid): mod.extract(write_manifest(base, bad))
            (base/"chat-hotset.json").write_text("changed after manifest\n")
            with self.assertRaises(mod.Invalid): mod.extract(write_manifest(base, manifest))

    def test_verbose_cycle_audit_binds_ids_epochs_outcomes_and_integer_costs(self):
        for k in range(3):
            lines = log(k, capture=True)
            result = mod.audit_cycles("audit", lines, mod.parse_log(lines, k, "capture"), {1})
            self.assertEqual(result, dict(proposals=8, epochs=4, cycles=2, steady_cycles=1))
            for field, value in (("next_epoch", 99), ("seq", 2), ("accepted", k+1),
                                 ("proposal_invalid_depths", 1), ("k", (k+1) % 3)):
                bad = list(lines)
                index = next(i for i, line in enumerate(bad) if mod.CYCLE in line)
                cycle = json.loads(mod.payload(bad[index], mod.CYCLE)); cycle[field] = value
                bad[index] = mod.CYCLE+json.dumps(cycle)
                with self.subTest(k=k, field=field), self.assertRaises(mod.Invalid):
                    mod.audit_cycles("audit", bad, mod.parse_log(bad, k, "capture"), {1})
            bad = lines+[next(line for line in lines if mod.CYCLE in line)]
            with self.assertRaises(mod.Invalid):
                mod.audit_cycles("audit", bad, mod.parse_log(bad, k, "capture"), {1})

    def test_full_prompt_audits_are_recorded_but_never_price_costs(self):
        with tempfile.TemporaryDirectory(prefix="ds41-cost-test-") as tmp:
            base = Path(tmp); manifest = fixture(base)
            run = copy.deepcopy(manifest["runs"][2])
            run.update(run="audit2", scope="audit_only", logging="capture", log_level="debug",
                       arm="all-Ds-Db2-Dme2-Dpc", log="audit.log", results="audit.json",
                       requests=[dict(seq=1, frozen_run="original", frozen_seq=2)])
            (base/"audit.log").write_text("\n".join(log(2, ns=30_000_000, capture=True)))
            (base/"audit.json").write_text(json.dumps(dict(concurrency=1, requests=[dict(name="code-02", family="code", text="reply")])))
            manifest["runs"].append(run)
            result = mod.extract(write_manifest(base, manifest))
            self.assertEqual(result["families"]["code"]["c2_ms"], 3.)
            self.assertEqual(result["extraction"]["runs"][-1]["cycle_audit"]["steady_cycles"], 1)


if __name__ == "__main__":
    unittest.main()
