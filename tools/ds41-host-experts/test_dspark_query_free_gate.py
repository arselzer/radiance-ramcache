"""Fake receipt cases only; source prepared for the parent's serialized host check."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("query_free_gate", Path(__file__).with_name("dspark_query_free_gate.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

START = gate.START + "\nDSpark context pass deferred (test)\n"
FIRST = ("DSpark query-free first: seq=1 step=1 boundary=1 slot=0 card=0 query_rows=0 "
         "observer_rows=0 query_epochs=0 context_rows=40 wait=step_end\n")
TOTAL = ("DSpark query-free totals: opportunities=5 waits=5 wait_ns=35 context_rows=44 "
         "query_issues=0 query_rows=0 observers=0 query_epochs=0 publications=0 "
         "transition=2 no_opportunity=0 end_cancel=1 unclosed_boundary=0\n")
COST = ("DSpark query-free costs: label=C0NQ n=2 cycle_ns=200 wait_ns=14 emitted=2 mean_ms=0.000100; "
        "consecutive target-commit-bound pure-decode opportunities only; "
        "context retained, early wait retained, no query/observer/query readback\n")
GOOD = START + FIRST + TOTAL + COST


class Gate(unittest.TestCase):
    def bad(self, text):
        with self.assertRaises(gate.Invalid):
            gate.check_log(text, True, True)

    def test_good_startup_and_complete(self):
        gate.check_log(START, True)
        gate.check_log(GOOD, True, True)

    def test_off_has_no_nq_receipt(self):
        gate.check_log("ordinary server", False, True)
        with self.assertRaises(gate.Invalid): gate.check_log(GOOD, False, True)

    def test_missing_actual_use_or_final_report(self):
        for piece in (FIRST, TOTAL, COST): self.bad(GOOD.replace(piece, ""))

    def test_duplicate_reports(self):
        for piece in (gate.START, FIRST, TOTAL, COST): self.bad(GOOD + piece)

    def test_query_epoch_publication_or_observer_cannot_be_nonzero(self):
        for field in ("query_epochs", "query_issues", "query_rows", "observers", "publications"):
            self.bad(GOOD.replace(f"{field}=0", f"{field}=1"))

    def test_unclosed_or_fabricated_cost(self):
        for old, new in (("n=2", "n=0"), ("end_cancel=1", "end_cancel=0"),
                         ("waits=5", "waits=4"), ("emitted=2", "emitted=3"),
                         ("0.000100", "0.000101")):
            self.bad(GOOD.replace(old, new))

    def test_no_context_or_wrong_actual_owner(self):
        self.bad(GOOD.replace("context_rows=40", "context_rows=0"))
        self.bad(GOOD.replace("seq=1", "seq=0"))
        self.bad(GOOD.replace("card=0", "card=1"))
        self.bad(GOOD.replace("DSpark context pass deferred (test)", ""))

    def test_conflicting_policy_or_lender(self):
        for marker in ("DSpark prefix policy", "DSpark matched early totals:", "DSBANKLEND"):
            self.bad(GOOD + marker)


if __name__ == "__main__":
    unittest.main()
