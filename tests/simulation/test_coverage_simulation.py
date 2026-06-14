"""Smoke tests for the coverage simulator.

These tests run a very small grid (``n=200``, 2 replicates, 2 methods) so the
suite stays fast.  Their job is to catch regressions in the DGP / true-uplift
plumbing and the aggregation, not to validate empirical coverage in the
statistical sense (which would need ``R=500+`` replicates).
"""

import pandas as pd

from tests.simulation.coverage_simulation import (
    DGPParams,
    aggregate_records,
    generate_dataset,
    run_grid,
    run_replicate,
)


class TestDGP:
    """Tests for the data-generating process helpers."""

    def test_dataset_shape(self):
        """Generated dataset has the expected row count and columns."""
        df = generate_dataset(500, DGPParams(), seed=0)
        assert df.shape == (500, 5)
        assert set(df.columns) == {'S1', 'S2', 'F1', 'F2', 'Y'}

    def test_attribute_values_in_known_set(self):
        """Each attribute only takes values from its known categorical domain."""
        df = generate_dataset(300, DGPParams(), seed=0)
        assert set(df['S1'].unique()) <= {'0', '1'}
        assert set(df['S2'].unique()) <= {'A', 'B'}
        assert set(df['F1'].unique()) <= {'x', 'y'}
        assert set(df['F2'].unique()) <= {'u', 'v'}
        assert set(df['Y'].unique()) <= {'0', '1'}

    def test_seed_reproducibility(self):
        """Same seed produces an identical dataset."""
        a = generate_dataset(200, DGPParams(), seed=42)
        b = generate_dataset(200, DGPParams(), seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_prob_y_bounds(self):
        """Target probability stays strictly within (0, 1) for all combinations."""
        params = DGPParams()
        for s1 in ('0', '1'):
            for s2 in ('A', 'B'):
                for f1 in ('x', 'y'):
                    for f2 in ('u', 'v'):
                        p = params.prob_y(s1, s2, f1, f2)
                        assert 0.0 < p < 1.0


class TestRunReplicate:
    """Tests for a single coverage-simulation replicate."""

    def test_returns_records(self):
        """A replicate on n=300 yields at least one well-formed coverage record."""
        recs = run_replicate(
            n=300,
            replicate_seed=0,
            params=DGPParams(),
            n_bootstrap=40,
            n_mc=200,
            methods=('wald',),
        )
        assert recs, "expected at least one rule on n=300"
        rec = recs[0]
        assert rec.method == 'wald'
        assert isinstance(rec.true_uplift, float)
        assert isinstance(rec.covered, bool)


class TestRunGrid:
    """Tests for running the full simulation grid."""

    def test_grid_runs_and_aggregates(self):
        """Grid run returns records and a summary with coverage in [0, 1]."""
        recs_df, summary = run_grid(
            sample_sizes=[300],
            n_replicates=2,
            params=DGPParams(),
            n_bootstrap=40,
            n_mc=200,
            methods=('wald', 'bootstrap_percentile'),
        )
        assert isinstance(recs_df, pd.DataFrame)
        assert isinstance(summary, pd.DataFrame)
        if not summary.empty:
            assert set(summary['method'].unique()) <= {'wald', 'bootstrap_percentile'}
            # Coverage in [0, 1]
            assert ((summary['empirical_coverage'] >= 0) & (summary['empirical_coverage'] <= 1)).all()


class TestAggregateRecords:
    """Tests for aggregation of coverage records."""

    def test_empty_returns_empty_frame(self):
        """Aggregating an empty record list returns an empty DataFrame."""
        out = aggregate_records([])
        assert out.empty
