"""Tests for the _bin_scaling helpers and the from_spectrum classmethods."""

from __future__ import annotations

import pytest

from maldideepkit import (
    MaldiCNNClassifier,
    MaldiMLPClassifier,
    MaldiResNetClassifier,
    MaldiTransformerClassifier,
)
from maldideepkit._bin_scaling import (
    REFERENCE_BIN_WIDTH,
    REFERENCE_CONV_KERNEL,
    scale_odd_kernel,
)


class TestScaleOddKernel:
    @pytest.mark.parametrize(
        "bin_width,expected",
        [
            # Reference: kernel=7 at bin_width=3, so target = 7*3/bw = 21/bw
            (1, 21),  # round(21) = 21, odd
            (2, 11),  # round(10.5) = 10 -> next odd = 11
            (3, 7),  # reference point
            (5, 5),  # round(4.2) = 4 -> next odd = 5
            (6, 5),  # round(3.5) = 4 -> next odd = 5
            (7, 3),  # round(3.0) = 3
            (10, 3),  # round(2.1) = 2 -> clamp to min 3
            (100, 3),  # clamped to min
        ],
    )
    def test_table(self, bin_width, expected):
        assert scale_odd_kernel(bin_width) == expected

    def test_always_odd(self):
        for bw in range(1, 50):
            assert scale_odd_kernel(bw) % 2 == 1

    def test_always_at_least_min(self):
        for bw in range(1, 50):
            assert scale_odd_kernel(bw, min_kernel=5) >= 5

    def test_custom_reference_kernel(self):
        # reference_kernel=13 at bin_width=3 -> 13 (ref bw == target bw)
        assert scale_odd_kernel(3, reference_kernel=13) == 13
        # reference_kernel=13 at bin_width=6 -> 13*3/6 = 6.5 -> 6 -> next odd = 7
        assert scale_odd_kernel(6, reference_kernel=13) == 7


class TestFromSpectrumSemanticSplit:
    """The factory's behaviour under the semantic split."""

    def test_mlp_forwards_input_dim_only(self):
        clf = MaldiMLPClassifier.from_spectrum(bin_width=6, input_dim=3000)
        assert clf.input_dim == 3000
        # MLP has no architectural knob to scale; hidden_dim stays default.
        assert clf.hidden_dim == MaldiMLPClassifier().hidden_dim

    def test_cnn_kernel_driven_by_bin_width(self):
        # Same bin_width, different input_dim -> same kernel
        a = MaldiCNNClassifier.from_spectrum(bin_width=3, input_dim=6000)
        b = MaldiCNNClassifier.from_spectrum(bin_width=3, input_dim=2000)
        assert a.kernel_size == b.kernel_size == 7

        # Different bin_width, same input_dim -> different kernel
        c = MaldiCNNClassifier.from_spectrum(bin_width=6, input_dim=6000)
        assert c.kernel_size == scale_odd_kernel(6)  # = 5

    def test_resnet_stem_driven_by_bin_width(self):
        a = MaldiResNetClassifier.from_spectrum(bin_width=3, input_dim=6000)
        b = MaldiResNetClassifier.from_spectrum(bin_width=3, input_dim=2000)
        assert a.stem_kernel_size == b.stem_kernel_size == 7

        c = MaldiResNetClassifier.from_spectrum(bin_width=10, input_dim=6000)
        assert c.stem_kernel_size == scale_odd_kernel(10)

    def test_resnet_from_spectrum_forces_peak_friendly_stem(self):
        # from_spectrum sets stem_stride=1 and use_stem_pool=False regardless
        # of bin width, so the peak-friendly configuration is always used.
        for bw in (1, 3, 6, 10):
            clf = MaldiResNetClassifier.from_spectrum(bin_width=bw, input_dim=3000)
            assert clf.stem_stride == 1
            assert clf.use_stem_pool is False

    def test_resnet_from_spectrum_overrides_win(self):
        # User can still opt into the literal ResNet-18 stem through overrides.
        clf = MaldiResNetClassifier.from_spectrum(
            bin_width=3, input_dim=6000, stem_stride=2, use_stem_pool=True
        )
        assert clf.stem_stride == 2
        assert clf.use_stem_pool is True

    def test_transformer_is_scale_agnostic(self):
        """Transformer's ``from_spectrum`` forwards ``input_dim`` and leaves
        architectural knobs alone - the learned positional embedding sizes
        itself to whatever token count the patch embedding produces, so no
        per-layout scaling is needed. Same rationale as the MLP."""
        a = MaldiTransformerClassifier.from_spectrum(bin_width=3, input_dim=6000)
        b = MaldiTransformerClassifier.from_spectrum(bin_width=6, input_dim=3000)
        c = MaldiTransformerClassifier.from_spectrum(bin_width=1, input_dim=12000)
        assert a.patch_size == b.patch_size == c.patch_size == 4
        assert a.embed_dim == b.embed_dim == c.embed_dim
        assert a.input_dim == 6000
        assert b.input_dim == 3000
        assert c.input_dim == 12000

    def test_override_wins(self):
        clf = MaldiCNNClassifier.from_spectrum(
            bin_width=6, input_dim=3000, kernel_size=9
        )
        assert clf.kernel_size == 9

    def test_extra_kwargs_propagate(self):
        clf = MaldiCNNClassifier.from_spectrum(
            bin_width=3, input_dim=6000, random_state=42, epochs=5
        )
        assert clf.random_state == 42
        assert clf.epochs == 5

    def test_reference_constants_expose(self):
        # Downstream docs reference these; guard against accidental renames.
        assert REFERENCE_BIN_WIDTH == 3
        assert REFERENCE_CONV_KERNEL == 7


class TestCNNScalarOrTuple:
    def test_scalar_kernel_size_still_works(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiCNNClassifier(
            channels=(8, 16),
            kernel_size=5,
            pool_size=2,
            head_dim=8,
            epochs=1,
            batch_size=8,
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    def test_tuple_kernel_size(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiCNNClassifier(
            channels=(8, 16),
            kernel_size=(7, 3),
            pool_size=2,
            head_dim=8,
            epochs=1,
            batch_size=8,
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    def test_tuple_pool_size(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiCNNClassifier(
            channels=(8, 16),
            kernel_size=3,
            pool_size=(2, 4),
            head_dim=8,
            epochs=1,
            batch_size=8,
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    def test_kernel_size_length_mismatch_raises(self):
        from maldideepkit.cnn.cnn import SpectralCNN1D

        with pytest.raises(ValueError, match="length"):
            SpectralCNN1D(
                input_dim=256,
                channels=(8, 16, 32),
                kernel_size=(5, 5),  # wrong length
            )


class TestResNetExposedStem:
    def test_stem_knobs_present(self):
        clf = MaldiResNetClassifier()
        # Peak-friendly defaults (see class docstring).
        assert clf.stem_kernel_size == 7
        assert clf.stem_stride == 1
        assert clf.block_kernel_size == 7
        assert clf.use_stem_pool is False

    def test_custom_stem(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiResNetClassifier(
            stem_channels=8,
            stage_channels=(8, 16),
            blocks_per_stage=(1, 1),
            stem_kernel_size=11,
            stem_stride=2,
            block_kernel_size=5,
            use_stem_pool=True,
            epochs=1,
            batch_size=8,
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)
        assert clf.use_stem_pool is True

    def test_block_kernel_matches_in_module(self):
        from maldideepkit.resnet.resnet import BasicBlock1D

        block = BasicBlock1D(4, 8, kernel_size=5)
        # Conv1d kernel_size is stored as a tuple.
        assert block.conv1.kernel_size == (5,)
        assert block.conv2.kernel_size == (5,)
