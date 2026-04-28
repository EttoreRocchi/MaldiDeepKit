"""Tests for the SAM optimizer wrapper."""

from __future__ import annotations

import pytest
import torch

from maldideepkit import MaldiMLPClassifier
from maldideepkit.utils import SAMOptimizer


class TestSAMOptimizer:
    def test_rejects_non_positive_rho(self):
        model = torch.nn.Linear(4, 2)
        with pytest.raises(ValueError, match="rho"):
            SAMOptimizer(model.parameters(), torch.optim.SGD, rho=0.0, lr=1e-3)

    def test_step_unsupported(self):
        model = torch.nn.Linear(4, 2)
        opt = SAMOptimizer(model.parameters(), torch.optim.SGD, rho=0.1, lr=1e-3)
        with pytest.raises(RuntimeError, match="two-pass"):
            opt.step()

    def test_first_step_then_second_step_updates_weights(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2)
        w_initial = model.weight.detach().clone()
        opt = SAMOptimizer(model.parameters(), torch.optim.SGD, rho=0.1, lr=1e-2)
        criterion = torch.nn.CrossEntropyLoss()
        X = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))

        opt.zero_grad()
        criterion(model(X), y).backward()
        opt.first_step(zero_grad=True)

        criterion(model(X), y).backward()
        opt.second_step(zero_grad=True)

        w_final = model.weight.detach().clone()
        assert not torch.allclose(w_initial, w_final)


class TestSAMClassifier:
    def test_default_is_off(self):
        clf = MaldiMLPClassifier()
        assert clf.use_sam is False

    def test_sam_fit_runs(self, synthetic_binary):
        X, y = synthetic_binary
        clf = MaldiMLPClassifier(
            hidden_dim=8,
            head_dims=(4,),
            epochs=2,
            batch_size=8,
            use_sam=True,
            sam_rho=0.05,
            random_state=0,
        ).fit(X, y)
        assert clf.predict(X).shape == (len(X),)
