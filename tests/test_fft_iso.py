"""The rewritten fft_power_iso must match the previous full-spectrum
implementation, which computed a complex FFT of the whole lattice and
binned shells with np.histogram."""
import numpy as np
import pytest

from dhybridrpy.data import fft_power_iso


def reference_spectrum(data, box_lengths, normalize=False):
    data = data.astype(np.float64)
    shape = data.shape
    deltas = [length / n for length, n in zip(box_lengths, shape)]
    fft_data = np.fft.fftn(data)
    power = np.abs(fft_data) ** 2 * np.prod(deltas) / np.prod(shape)
    freqs = [
        np.fft.fftfreq(n, d=d) * 2 * np.pi for n, d in zip(shape, deltas)
    ]
    if len(shape) == 1:
        k_magnitude = np.abs(freqs[0])
    else:
        grids = np.meshgrid(*freqs, indexing="ij")
        k_magnitude = np.sqrt(sum(g**2 for g in grids))
    k_max = min(np.abs(f).max() for f in freqs)
    dk = 2 * np.pi / max(box_lengths)
    k_bins = np.arange(0, k_max + dk, dk)
    binned, _ = np.histogram(
        k_magnitude.ravel(), bins=k_bins, weights=power.ravel()
    )
    if normalize:
        counts, _ = np.histogram(k_magnitude.ravel(), bins=k_bins)
        binned = np.divide(
            binned, counts, out=np.zeros_like(binned), where=counts > 0
        )
    return 0.5 * (k_bins[:-1] + k_bins[1:]), binned


CASES = [
    ((16,), (10.0,)),
    ((17,), (7.3,)),
    ((16, 12), (10.0, 7.0)),
    ((17, 13), (9.1, 6.4)),
    ((12, 12, 12), (5.0, 5.0, 5.0)),   # cubic: corner modes drop
    ((16, 12, 10), (10.0, 7.0, 5.5)),  # anisotropic box and grid
    ((15, 13, 11), (8.0, 8.0, 8.0)),   # all-odd sizes
]


@pytest.mark.parametrize("shape,box", CASES)
@pytest.mark.parametrize("normalize", [False, True])
def test_matches_full_spectrum_reference(shape, box, normalize):
    rng = np.random.default_rng(sum(shape))
    data = (1.0 + 0.5 * rng.standard_normal(shape)).astype(np.float64)
    k_ref, p_ref = reference_spectrum(data, box, normalize)
    k_new, p_new = fft_power_iso(data, *box, normalize=normalize)
    np.testing.assert_array_equal(k_new, k_ref)
    np.testing.assert_allclose(p_new, p_ref, rtol=1e-9, atol=1e-14)


@pytest.mark.parametrize("shape", [(1,), (1, 8), (8, 1), (4, 4, 1), (1, 4, 4)])
@pytest.mark.parametrize("normalize", [False, True])
def test_size_one_axis_returns_empty(shape, normalize):
    data = np.ones(shape)
    box = tuple(float(n) for n in shape)
    k, p = fft_power_iso(data, *box, normalize=normalize)
    assert len(k) == 0 and len(p) == 0


@pytest.mark.parametrize("shape,box", CASES)
def test_float32_input_close_to_float64(shape, box):
    rng = np.random.default_rng(1 + sum(shape))
    data = (1.0 + 0.5 * rng.standard_normal(shape)).astype(np.float32)
    _, p32 = fft_power_iso(data, *box)
    _, p64 = fft_power_iso(data.astype(np.float64), *box)
    np.testing.assert_allclose(p32, p64, rtol=1e-4)
