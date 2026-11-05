"""
Tests for v12.truth — the checks that keep hardware readings honest.

Every test here corresponds to a defect that actually shipped in this module, or
to the structural guard that would have caught it. That is the standard: a test
that only confirms the happy path would not have found any of these, because
every wrong value the module produced was perfectly well-formed.

The bugs, for reference:
  * sm_count read 16384 instead of 128, from a correctly-named function that
    returns a different quantity than its name suggests.
  * Every FLOP rate was half the real value, so every ceiling was halved.
  * A live clock was used where a maximum was required.
All three produced numbers that looked reasonable. None raised an error.
"""
from __future__ import annotations

import unittest

from v12 import truth as T


# ---------------------------------------------------------------------------
# helpers — synthetic readers, so no test here needs a GPU
# ---------------------------------------------------------------------------
def _readers(*specs):
    """(name, {field: value}) pairs -> the shape _merge consumes."""
    out = []
    for name, vals in specs:
        out.append((name, dict(vals), {k: name for k in vals}))
    return out


class TestAgreement(unittest.TestCase):
    def test_identities_must_match_exactly(self):
        self.assertTrue(T._agrees("sm_count", 128, 128))
        self.assertFalse(T._agrees("sm_count", 128, 129))

    def test_a_near_miss_on_a_core_count_is_a_contradiction(self):
        # 2% of 16384 is 327 cores. That is not a rounding difference.
        self.assertFalse(T._agrees("cuda_cores", 16384, 16057))
        self.assertFalse(T._agrees("mem_bus_width_bits", 384, 376))

    def test_strings_compare_case_and_whitespace_insensitively(self):
        self.assertTrue(T._agrees("name", "NVIDIA GeForce RTX 4090", "nvidia geforce rtx 4090 "))
        self.assertFalse(T._agrees("name", "RTX 4090", "RTX 4080"))

    def test_measured_quantities_tolerate_two_percent(self):
        # 3105 vs 3050 is 1.8% — the same clock, reported by two different readers.
        self.assertTrue(T._agrees("clock_sm_max_mhz", 3105, 3050))
        # 3105 vs 2520 is 19% — two different quantities, not a rounding difference.
        self.assertFalse(T._agrees("clock_sm_max_mhz", 3105, 2520))

    def test_bools_compare_exactly(self):
        self.assertTrue(T._agrees("hags_enabled", True, True))
        self.assertFalse(T._agrees("hags_enabled", True, False))


class TestContradictionIsNotResolved(unittest.TestCase):
    """Two readers disagreeing is the most valuable signal on the machine.
    Precedence must not be allowed to pick a winner and hide it."""

    def test_a_contradicted_field_is_dropped_not_arbitrated(self):
        m = T._merge(_readers(
            ("cuda", {"sm_count": 128}),
            ("nvml", {"sm_count": 16384}),
        ))
        self.assertNotIn("sm_count", m.values,
                         "precedence picked a winner instead of reporting the conflict")
        self.assertEqual(len(m.disagreements), 1)
        self.assertIn("16384", m.disagreements[0])
        self.assertIn("128", m.disagreements[0])

    def test_agreeing_readers_produce_a_value(self):
        m = T._merge(_readers(
            ("nvml", {"clock_sm_max_mhz": 3105}),
            ("smi", {"clock_sm_max_mhz": 3100}),
        ))
        self.assertEqual(m.values["clock_sm_max_mhz"], 3105)
        self.assertEqual(m.source["clock_sm_max_mhz"], "nvml")
        self.assertEqual(m.disagreements, [])

    def test_precedence_order_is_respected_when_readers_agree(self):
        m = T._merge(_readers(
            ("smi", {"driver_version": "616.92"}),
            ("nvml", {"driver_version": "616.92"}),
        ))
        self.assertEqual(m.source["driver_version"], "nvml")


class TestPhysicalBands(unittest.TestCase):
    """The check with no second reader to blame — it works from one value."""

    def test_an_impossible_sm_count_is_discarded(self):
        # No NVIDIA GPU has ever had 16384 SMs. This is the actual number that
        # nvmlDeviceGetNumGpuCores returned for a card with 128 of them.
        m = T._merge(_readers(("nvml", {"sm_count": 16384})))
        self.assertNotIn("sm_count", m.values)
        self.assertTrue(any("outside the possible range" in e for e in m.errors),
                        f"no band error raised; errors={m.errors}")

    def test_a_plausible_sm_count_survives(self):
        m = T._merge(_readers(("cuda", {"sm_count": 128})))
        self.assertEqual(m.values["sm_count"], 128)

    def test_a_bus_width_that_is_not_a_multiple_of_32_is_discarded(self):
        # Attribute 38 returned 75497472 on this driver. That is not a bus width,
        # and the shape check catches it without needing to know the right ID.
        m = T._merge(_readers(("cuda", {"mem_bus_width_bits": 75497472})))
        self.assertNotIn("mem_bus_width_bits", m.values)
        self.assertTrue(any("multiple of 32" in e or "outside the possible range" in e
                            for e in m.errors), m.errors)

    def test_a_real_bus_width_survives(self):
        m = T._merge(_readers(("nvml", {"mem_bus_width_bits": 384})))
        self.assertEqual(m.values["mem_bus_width_bits"], 384)

    def test_a_live_clock_outside_its_band_is_discarded(self):
        m = T._merge(_readers(("nvml", {"clock_sm_max_mhz": 99})))
        self.assertNotIn("clock_sm_max_mhz", m.values)


class TestLanesPerSmIdentity(unittest.TestCase):
    """The cross-check with teeth: it compares two measurements of the same
    silicon, so it needs no second reader and no datasheet."""

    def test_cores_over_sms_must_be_a_real_lanes_per_sm_figure(self):
        m = T._merge(_readers(("nvml", {
            "cuda_cores": 16384,
            "sm_count": 128,
        })))
        self.assertEqual(m.values["cuda_cores"], 16384)
        self.assertEqual(m.values["sm_count"], 128)
        # 16384 / 128 == 128 lanes per SM, which is Ada's real figure.
        self.assertFalse(any("lanes per SM" in e for e in m.errors), m.errors)

    def test_the_historical_value_is_caught_by_the_band(self):
        """16384 SMs is impossible, so the range check gets there first."""
        m = T._merge(_readers(("nvml", {"cuda_cores": 16384, "sm_count": 16384})))
        self.assertNotIn("sm_count", m.values)
        self.assertTrue(any("outside the possible range" in e for e in m.errors), m.errors)

    def test_a_plausible_but_wrong_sm_count_is_caught_only_by_the_identity(self):
        """The case the band cannot reach — and the reason the identity exists.

        512 SMs passes every range check and is still four times this card's real
        128. Paired with the true 16384 cores it implies 32 lanes per SM, which
        no NVIDIA SM has ever had.
        """
        m = T._merge(_readers(("nvml", {"cuda_cores": 16384, "sm_count": 512})))
        self.assertTrue(any("lanes per SM" in e for e in m.errors),
                        f"a plausible wrong SM count slipped through: {m.errors}")

    def test_an_identity_failure_publishes_neither_number(self):
        """There is no way to tell which of the two is wrong, and sm_count feeds
        every ceiling in this module. A missing ceiling is recoverable; a wrong
        one is not."""
        m = T._merge(_readers(("nvml", {"cuda_cores": 16384, "sm_count": 512})))
        self.assertNotIn("sm_count", m.values)
        self.assertNotIn("cuda_cores", m.values)
        # And the module says so rather than silently emptying.
        self.assertTrue(any("recoverable" in e for e in m.errors), m.errors)

    def test_cores_without_sms_is_reported_not_estimated(self):
        m = T._merge(_readers(("nvml", {"cuda_cores": 16384})))
        self.assertTrue(any("sm_count is not" in e for e in m.errors), m.errors)


class TestFlopTableClearsMeasurement(unittest.TestCase):
    """The regression test for the 2x error, in its honest form.

    The check asks whether reaching a *measured* throughput would need a clock the
    silicon does not have. That refutes the fp32 half of the old table
    decisively and — this is the part worth being precise about — cannot refute
    the fp16 half on its own. Claiming otherwise would credit the check with more
    than it tests, which is the same failure as a number that looks more
    authoritative than it is.
    """

    @staticmethod
    def _required_mhz(rate: float, tflops: float, sms: int = 128) -> float:
        return (tflops * 1e12 / (rate * sms)) / 1e6

    def test_the_old_fp32_rate_needs_an_impossible_clock(self):
        self.assertAlmostEqual(self._required_mhz(128.0, 88.20), 5385, delta=5)
        self.assertGreater(self._required_mhz(128.0, 88.20), 3105,
                           "an fp32 rate of 128/SM/clk must require more clock than "
                           "the card has, or the check does not refute it")

    def test_the_current_fp32_rate_needs_a_clock_the_card_reaches(self):
        self.assertLess(self._required_mhz(256.0, 88.20), 3105)

    def test_the_check_refutes_the_old_fp32_half_in_the_real_merge(self):
        original = T.FLOP_PER_SM_CLK["fp32_cuda"]
        T.FLOP_PER_SM_CLK["fp32_cuda"] = 128.0
        try:
            m = T._merge(_readers(("cuda", {
                "sm_count": 128, "clock_sm_rated_mhz": 2520, "clock_sm_max_mhz": 3105,
            })))
        finally:
            T.FLOP_PER_SM_CLK["fp32_cuda"] = original
        self.assertTrue(any("cannot be right" in e for e in m.errors), m.errors)
        self.assertTrue(any("fp32_cuda" in e for e in m.errors), m.errors)

    def test_the_check_does_not_pretend_to_refute_the_old_fp16_rate(self):
        """Honesty about scope. 512/SM/clk needs 2521 MHz, which this card holds,
        so the measurement alone does NOT refute it — the ratio test below is
        what pins fp16."""
        self.assertLess(self._required_mhz(512.0, 165.26), 3105,
                        "if this ever becomes impossible, the docstring's claim "
                        "about which half is refuted by measurement is wrong")

    def test_the_current_table_passes_in_the_real_merge(self):
        m = T._merge(_readers(("cuda", {
            "sm_count": 128, "clock_sm_rated_mhz": 2520, "clock_sm_max_mhz": 3105,
        })))
        self.assertFalse(any("cannot be right" in e for e in m.errors),
                         f"a healthy table was refuted: {m.errors}")

    def test_the_check_uses_the_max_clock_not_the_rated_one(self):
        """A card that boosts above its rated clock must not trip the check.

        fp32 measured 88.20 TFLOPS needs ~2690 MHz on 128 SMs, which is 6.8%
        above the rated 2520 MHz. Judging the table at the rated clock alone
        would fire on a perfectly healthy GPU — a false alarm is its own bug.
        """
        m = T._merge(_readers(("cuda", {
            "sm_count": 128, "clock_sm_rated_mhz": 2520, "clock_sm_max_mhz": 3105,
        })))
        self.assertFalse(any("fp32_cuda" in e for e in m.errors), m.errors)
        # And with no max clock known, headroom is assumed rather than ignored.
        m2 = T._merge(_readers(("cuda", {"sm_count": 128, "clock_sm_rated_mhz": 2520})))
        self.assertFalse(any("fp32_cuda" in e for e in m2.errors), m2.errors)

    def test_a_rated_clock_above_the_maximum_is_reported(self):
        m = T._merge(_readers(
            ("cuda", {"clock_sm_rated_mhz": 4000}),
            ("nvml", {"clock_sm_max_mhz": 3105}),
        ))
        self.assertTrue(any("not a clock rate" in e for e in m.errors), m.errors)


class TestMaxIsNeverConfusedWithLive(unittest.TestCase):
    """Bug 3: a live memory clock fed the bandwidth figure and understated it
    26x at idle, while looking entirely reasonable."""

    def _t(self, **kw):
        t = T.GpuTruth()
        for k, v in kw.items():
            setattr(t, k, v)
        return t

    def test_bandwidth_ignores_the_live_clock_entirely(self):
        base = dict(mem_bus_width_bits=384, clock_mem_max_mhz=10501)
        idle = self._t(**base, clock_mem_live_mhz=405)
        busy = self._t(**base, clock_mem_live_mhz=10501)
        self.assertEqual(idle.mem_bandwidth_gbs, busy.mem_bandwidth_gbs,
                         "the live memory clock leaked into a peak calculation")

    def test_bandwidth_matches_the_published_figure(self):
        t = self._t(mem_bus_width_bits=384, clock_mem_max_mhz=10501)
        self.assertAlmostEqual(t.mem_bandwidth_gbs, 1008.1, delta=0.2)

    def test_bandwidth_is_zero_rather_than_guessed_without_a_max_clock(self):
        t = self._t(mem_bus_width_bits=384, clock_mem_live_mhz=10501)
        self.assertEqual(t.mem_bandwidth_gbs, 0.0)

    def test_peak_uses_the_rated_clock_not_the_driver_maximum(self):
        t = self._t(sm_count=128, clock_sm_rated_mhz=2520, clock_sm_max_mhz=3105)
        self.assertEqual(t.sm_clock_for_peak_mhz, 2520,
                         "a ceiling computed at a clock no sustained kernel holds "
                         "makes every kernel look worse than it is")
        self.assertAlmostEqual(t.peak_tflops("fp16_tensor"), 330.3, delta=0.2)

    def test_peak_falls_back_to_the_max_clock_only_when_rated_is_unknown(self):
        t = self._t(sm_count=128, clock_sm_max_mhz=3105)
        self.assertEqual(t.sm_clock_for_peak_mhz, 3105)

    def test_peak_is_zero_rather_than_guessed_when_incomplete(self):
        self.assertEqual(self._t(clock_sm_rated_mhz=2520).peak_tflops("fp16_tensor"), 0.0)
        self.assertEqual(self._t(sm_count=128).peak_tflops("fp16_tensor"), 0.0)

    def test_an_unknown_precision_returns_zero(self):
        t = self._t(sm_count=128, clock_sm_rated_mhz=2520)
        self.assertEqual(t.peak_tflops("fp4_tensor"), 0.0)

    def test_ridge_point_needs_both_halves(self):
        t = self._t(sm_count=128, clock_sm_rated_mhz=2520, mem_bus_width_bits=384,
                    clock_mem_max_mhz=10501)
        self.assertAlmostEqual(t.ridge_point("fp16_tensor"), 327.6, delta=1.0)
        self.assertEqual(self._t(sm_count=128, clock_sm_rated_mhz=2520).ridge_point(), 0.0)


class TestCompleteness(unittest.TestCase):
    def _t(self, **kw):
        t = T.GpuTruth()
        for k, v in kw.items():
            setattr(t, k, v)
        return t

    def test_incomplete_is_incomplete(self):
        self.assertFalse(self._t().complete)
        self.assertTrue(self._t(sm_count=128, clock_sm_rated_mhz=2520,
                                mem_bus_width_bits=384, clock_mem_max_mhz=10501).complete)

    def test_an_incomplete_summary_names_what_is_missing(self):
        # "0 SM @ 0 MHz" was the old failure: a zero that reads as a measurement.
        s = self._t(name="RTX 4090").summary
        self.assertIn("incomplete", s)
        self.assertIn("SM count", s)
        self.assertNotIn("0 SM @ 0 MHz", s)

    def test_a_complete_summary_carries_the_numbers(self):
        s = self._t(name="NVIDIA GeForce RTX 4090", driver_version="616.92",
                    compute_capability="8.9", sm_count=128, clock_sm_rated_mhz=2520,
                    vram_total_bytes=24 * 2**30, mem_bus_width_bits=384,
                    clock_mem_max_mhz=10501).summary
        self.assertIn("128 SM", s)
        self.assertIn("1008 GB/s", s)


class TestTableShape(unittest.TestCase):
    """The constants themselves, checked for the relationships that must hold."""

    def test_ada_rates_are_the_dense_ratios(self):
        # fp32 : tf32 : fp16 : fp8 = 1 : 2 : 4 : 8. Ada's tensor cores double per
        # precision step; the old table was half of this at every level.
        r = T.FLOP_PER_SM_CLK
        self.assertEqual(r["fp32_cuda"] * 2, r["tf32_tensor"])
        self.assertEqual(r["tf32_tensor"] * 2, r["fp16_tensor"])
        self.assertEqual(r["fp16_tensor"] * 2, r["fp8_tensor"])
        self.assertEqual(r["fp8_tensor"], r["int8_tensor"])

    def test_fp32_is_lanes_times_two(self):
        self.assertEqual(T.FLOP_PER_SM_CLK["fp32_cuda"], 128 * 2)

    def test_measured_floor_clocks_clear_the_implied_ceiling(self):
        """Every floor must be reachable at some clock the card has."""
        for kind, (floor, note) in T.MEASURED_FLOOR_TFLOPS.items():
            required = (floor * 1e12 / (T.FLOP_PER_SM_CLK[kind] * 128)) / 1e6
            self.assertLess(required, 3105,
                            f"{kind}: reaching {floor} TFLOPS at the table's rate "
                            f"would need {required:.0f} MHz, above this card's maximum")
            self.assertTrue(note, f"{kind}: a floor with no provenance")

    def test_the_fp16_and_fp8_rates_are_pinned_by_ratio_not_by_measurement(self):
        """Being explicit about the weaker evidence for the upper two rows.

        Only fp32 and fp16 have measurements on record. fp8's rate has none, and
        it is 2x fp16 purely because Ada's tensor cores double per precision
        step. That is a structural claim, not a measured one, and the report says
        so out loud rather than printing it as though it were tested.
        """
        self.assertNotIn("fp8_tensor", T.MEASURED_FLOOR_TFLOPS)
        self.assertEqual(T.FLOP_PER_SM_CLK["fp8_tensor"],
                         2 * T.FLOP_PER_SM_CLK["fp16_tensor"])

    def test_ada_has_no_thread_block_clusters_or_fp4(self):
        # Recorded here because the same false claim lived in the docs AND in
        # acceleration/cutlass_integration.py. sm_89 is not sm_90.
        self.assertNotIn("fp4_tensor", T.FLOP_PER_SM_CLK)
        self.assertTrue(T.BACKEND_REACHABLE["cutlass_cpp"])
        self.assertFalse(T.BACKEND_REACHABLE["cute_dsl"])


class TestReaderMissIsNotAFieldFailure(unittest.TestCase):
    """The live failure, reproduced without a GPU.

    On this box the NVML binding has no nvmlDeviceGetAttribute, so NVML cannot
    answer sm_count while the CUDA driver can. `.errors` promises fields nobody
    could establish; sm_count was established. The NVML miss is still worth
    keeping — it means sm_count had no second reader — so it moves, it does not
    vanish.
    """

    MISS = "nvml:sm_count: nvmlDeviceGetAttribute absent from this binding"

    def _truth(self, nvml, cuda, smi=({}, {}, [])):
        from unittest import mock
        with mock.patch.object(T, "_read_nvml", lambda: nvml), \
             mock.patch.object(T, "_read_cuda", lambda: cuda), \
             mock.patch.object(T, "_read_smi", lambda: smi), \
             mock.patch.object(T, "_hags_enabled", lambda: None), \
             mock.patch.object(T, "_resizable_bar", lambda: None):
            return T.gpu_truth()

    def test_a_field_another_reader_supplied_is_not_reported_as_failed(self):
        t = self._truth(nvml=({}, {}, [self.MISS]),
                        cuda=({"sm_count": 128}, {"sm_count": "cuda:attr16"}, []))
        self.assertEqual(t.sm_count, 128)
        self.assertEqual(t.source["sm_count"], "cuda:attr16")
        self.assertFalse(any(":sm_count:" in e for e in t.errors), t.errors)
        self.assertIn(self.MISS, t.notes, "the miss was dropped rather than recorded")

    def test_the_miss_stays_an_error_when_nobody_supplied_the_field(self):
        t = self._truth(nvml=({}, {}, [self.MISS]), cuda=({}, {}, []))
        self.assertIn(self.MISS, t.errors)
        self.assertTrue(any("SM count unread" in e for e in t.errors), t.errors)
        self.assertNotIn(self.MISS, t.notes)

    def test_the_miss_stays_an_error_when_the_other_value_was_discarded(self):
        # CUDA answers, but with an impossible number the band throws away. The
        # field is unread, so the NVML miss is part of why — not a footnote.
        t = self._truth(nvml=({}, {}, [self.MISS]),
                        cuda=({"sm_count": 16384}, {"sm_count": "cuda:attr16"}, []))
        self.assertEqual(t.sm_count, 0)
        self.assertIn(self.MISS, t.errors)
        self.assertNotIn(self.MISS, t.notes)

    def test_a_reader_level_failure_is_always_an_error(self):
        t = self._truth(nvml=({}, {}, ["nvml: nvmlInit failed: boom"]),
                        cuda=({"sm_count": 128}, {"sm_count": "cuda:attr16"}, []))
        self.assertIn("nvml: nvmlInit failed: boom", t.errors)

    def test_a_contradiction_is_still_refused(self):
        t = self._truth(nvml=({"sm_count": 512}, {"sm_count": "nvml:attr16"}, []),
                        cuda=({"sm_count": 128}, {"sm_count": "cuda:attr16"}, []))
        self.assertEqual(t.sm_count, 0)
        self.assertEqual(len(t.disagreements), 1)


class TestLiveReads(unittest.TestCase):
    """The real machine, when there is one. Skipped rather than faked."""

    def test_the_three_readers_are_independent(self):
        """At least one reader must answer without NVML, because that is the
        interpreter the launcher used to pick."""
        vals, _src, _errs = T._read_smi()
        if not vals:
            self.skipTest("nvidia-smi unavailable on this machine")
        self.assertIn("name", vals)
        self.assertNotIn("sm_count", vals,
                         "nvidia-smi has no SM-count query; it must not invent one")

    def test_a_live_read_is_self_consistent(self):
        t = T.gpu_truth()
        if not t.readers:
            self.skipTest("no reader answered")
        if t.sm_count and t.cuda_cores:
            self.assertIn(t.cuda_cores / t.sm_count, T.LANES_PER_SM_PERMITTED)
        if t.complete:
            self.assertGreater(t.peak_tflops("fp16_tensor"), 0.0)
            self.assertGreater(t.mem_bandwidth_gbs, 0.0)
            # A reader being absent is not an error about a value; every error
            # that names a field must not name one we actually read.
            for e in t.errors:
                for field_name in t.values_read():
                    self.assertNotIn(f":{field_name}:", e,
                                     f"{field_name} was read but is also reported as "
                                     f"a failure: {e}")

    def test_a_live_read_never_requires_a_clock_the_card_lacks(self):
        t = T.gpu_truth()
        if not t.complete:
            self.skipTest("incomplete read; nothing to check")
        for kind, (floor, _note) in T.MEASURED_FLOOR_TFLOPS.items():
            required = (floor * 1e12 / (T.FLOP_PER_SM_CLK[kind] * t.sm_count)) / 1e6
            self.assertLess(required, t.clock_sm_max_mhz or t.clock_sm_rated_mhz,
                            f"reaching the measured {kind} throughput would need "
                            f"{required:.0f} MHz; this card cannot do that, so the "
                            f"rate is wrong")

    def test_json_output_round_trips(self):
        import json
        d = json.loads(T.describe(as_json=True))
        self.assertIn("gpu", d)
        self.assertIn("roofline", d)
        self.assertIn("backends", d)
        # Every field a reader might branch on is present even when unreadable.
        for key in ("sm_count", "clock_sm_max_mhz", "clock_sm_rated_mhz",
                    "clock_mem_max_mhz", "clock_mem_live_mhz", "disagreements"):
            self.assertIn(key, d["gpu"], f"{key} missing from the JSON view")


if __name__ == "__main__":
    unittest.main()
