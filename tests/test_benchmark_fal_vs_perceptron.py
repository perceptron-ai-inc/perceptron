"""
Benchmark test comparing Fal vs Perceptron providers with Isaac 0.1.

This test ensures that the Perceptron provider is not statistically
significantly slower than the Fal provider for object detection tasks.
"""

import os
import time

import pytest
from scipy import stats

from cookbook.utils import cookbook_asset
from perceptron import detect
from perceptron import config as cfg

pytestmark = pytest.mark.integration

_PERCEPTRON_API_KEY = os.getenv("PERCEPTRON_API_KEY")
_FAL_KEY = os.getenv("FAL_KEY")
_PERCEPTRON_BASE_URL = os.getenv("PERCEPTRON_BASE_URL")
_TARGET_IMAGE = cookbook_asset("in-context-learning", "multi", "cat_dog_input.png")

requires_both_keys = pytest.mark.skipif(
    not (_PERCEPTRON_API_KEY and _FAL_KEY),
    reason="Both PERCEPTRON_API_KEY and FAL_KEY must be set to run benchmark tests.",
)


def _run_detection(provider: str, api_key: str, base_url: str | None = None) -> None:
    """Run a simple object detection task."""
    cfg_kwargs = {
        "provider": provider,
        "api_key": api_key,
        "model": "isaac-0.1",
    }
    # Only set base_url if explicitly provided to avoid env var override
    if base_url is not None:
        cfg_kwargs["base_url"] = base_url

    with cfg(**cfg_kwargs):
        result = detect(
            str(_TARGET_IMAGE),
            classes=["cat", "dog"],
            temperature=0.0,
            max_tokens=256,
        )
        # Ensure the request succeeded
        assert result.raw.get("choices"), f"Provider {provider} did not return choices"


@requires_both_keys
def test_benchmark_fal_detection(benchmark):
    """Benchmark Fal provider with Isaac 0.1 for object detection."""
    # Explicitly set Fal's base URL to prevent PERCEPTRON_BASE_URL override
    benchmark(_run_detection, "fal", _FAL_KEY, "https://fal.run")


@requires_both_keys
def test_benchmark_perceptron_detection(benchmark):
    """Benchmark Perceptron provider with Isaac 0.1 for object detection."""
    # Use PERCEPTRON_BASE_URL if set, otherwise use default
    benchmark(_run_detection, "perceptron", _PERCEPTRON_API_KEY, _PERCEPTRON_BASE_URL)


@requires_both_keys
def test_perceptron_not_significantly_slower_than_fal():
    """
    Statistical test to ensure Perceptron is not significantly slower than Fal.

    This test runs multiple iterations of object detection on both providers
    and performs a one-tailed t-test to verify that Perceptron's mean latency
    is not statistically significantly greater than Fal's at p < 0.05.

    Note: This test runs independently of pytest-benchmark to collect raw timing data.
    """
    # Number of samples for statistical analysis
    n_samples = 20

    print(f"\nCollecting {n_samples} samples from each provider...")

    # Collect timing data for Fal
    fal_times = []
    for i in range(n_samples):
        start = time.perf_counter()
        _run_detection("fal", _FAL_KEY, "https://fal.run")
        elapsed = time.perf_counter() - start
        fal_times.append(elapsed)
        print(f"  Fal sample {i+1}/{n_samples}: {elapsed:.4f}s")

    # Collect timing data for Perceptron
    perceptron_times = []
    for i in range(n_samples):
        start = time.perf_counter()
        _run_detection("perceptron", _PERCEPTRON_API_KEY, _PERCEPTRON_BASE_URL)
        elapsed = time.perf_counter() - start
        perceptron_times.append(elapsed)
        print(f"  Perceptron sample {i+1}/{n_samples}: {elapsed:.4f}s")

    # Calculate statistics
    fal_mean = sum(fal_times) / len(fal_times)
    perceptron_mean = sum(perceptron_times) / len(perceptron_times)

    # Perform one-tailed t-test
    # H0: Perceptron mean <= Fal mean
    # H1: Perceptron mean > Fal mean
    # We want to reject H1 (i.e., fail to reject H0)
    t_statistic, p_value = stats.ttest_ind(perceptron_times, fal_times, alternative="greater")

    # Print results for visibility
    print(f"\n{'='*60}")
    print(f"Benchmark Results:")
    print(f"{'='*60}")
    print(f"Fal mean latency:        {fal_mean:.4f}s")
    print(f"Perceptron mean latency: {perceptron_mean:.4f}s")
    print(f"Difference:              {perceptron_mean - fal_mean:+.4f}s ({((perceptron_mean / fal_mean - 1) * 100):+.2f}%)")
    print(f"T-statistic:             {t_statistic:.4f}")
    print(f"P-value (one-tailed):    {p_value:.4f}")
    print(f"{'='*60}")

    # Assert that Perceptron is not significantly slower (p >= 0.05)
    # If p < 0.05, we reject H0 and conclude Perceptron IS significantly slower
    assert (
        p_value >= 0.05
    ), f"Perceptron is statistically significantly slower than Fal (p={p_value:.4f} < 0.05, mean difference={perceptron_mean - fal_mean:.4f}s)"

    print(f"✓ Perceptron is not significantly slower than Fal (p={p_value:.4f} >= 0.05)")
