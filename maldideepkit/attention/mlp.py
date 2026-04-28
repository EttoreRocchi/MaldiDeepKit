"""MLP classifier with optional sigmoid-gated attention.

The architecture is a dense network with a learned per-feature gate on
the first hidden layer that doubles as an interpretable attention map
over the projected bin representation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.utils.validation import check_is_fitted
from torch import nn

from ..base.classifier import BaseSpectralClassifier
from ..base.data import _to_numpy


class SpectralAttentionMLP(nn.Module):
    """Projection + optional sigmoid-gated attention + deep MLP head.

    Parameters
    ----------
    input_dim : int
        Number of input bins. The first linear layer projects this down
        to :attr:`hidden_dim`.
    n_classes : int, default=2
        Number of output logits.
    hidden_dim : int, default=512
        Width of the projection layer and attention gate.
    head_dims : sequence of int, default=(256, 128)
        Widths of the hidden layers between the gated representation and
        the output logits.
    use_attention : bool, default=True
        If ``True``, apply a sigmoid-gated element-wise attention on the
        projected features. If ``False``, the model reduces to a plain
        MLP of the same depth.
    dropout_high : float, default=0.3
        Dropout applied after the projection and the first dense layer.
    dropout_low : float, default=0.2
        Dropout applied before the output logits.

    Attributes
    ----------
    last_attention : torch.Tensor or None
        Attention weights from the most recent forward pass
        (``(batch, hidden_dim)``). ``None`` when ``use_attention=False``.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int = 2,
        hidden_dim: int = 512,
        head_dims: tuple[int, ...] = (256, 128),
        use_attention: bool = True,
        dropout_high: float = 0.3,
        dropout_low: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_high),
        )
        if use_attention:
            self.attn: nn.Module = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()
            )
        else:
            self.attn = nn.Identity()

        head_layers: list[nn.Module] = []
        prev = hidden_dim
        for i, width in enumerate(head_dims):
            head_layers += [
                nn.Linear(prev, width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Dropout(dropout_high if i == 0 else dropout_low),
            ]
            prev = width
        head_layers.append(nn.Linear(prev, n_classes))
        self.head = nn.Sequential(*head_layers)

        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, input_dim)`` to ``(batch, n_classes)`` logits."""
        projected = self.proj(x)
        if self.use_attention:
            weights = self.attn(projected)
            self.last_attention = weights.detach()
            gated = projected * weights
        else:
            self.last_attention = None
            gated = projected
        return self.head(gated)


class MaldiMLPClassifier(BaseSpectralClassifier):
    """sklearn-compatible MLP classifier with optional attention gating.

    Parameters
    ----------
    hidden_dim : int, default=512
        Width of the projection and attention gate.
    head_dims : sequence of int, default=(256, 128)
        Widths of the hidden layers of the classification head.
    use_attention : bool, default=True
        Toggle the sigmoid-gated attention. When ``False``, the model
        is a plain MLP of the same depth.
    dropout_high : float, default=0.3
        Dropout after the projection and first head layer.
    dropout_low : float, default=0.2
        Dropout before the output logits.
    **kwargs
        Forwarded to :class:`~maldideepkit.base.classifier.BaseSpectralClassifier`:
        ``input_dim``, ``n_classes``, ``learning_rate``, ``batch_size``,
        ``epochs``, ``early_stopping_patience``, ``val_fraction``,
        ``standardize``, ``class_weight``, ``device``, ``random_state``,
        ``verbose``.

    Attributes
    ----------
    attention_weights_ : ndarray or None
        Attention weights from the last :meth:`fit` or :meth:`predict`
        forward pass. Shape ``(n_samples_last_call, hidden_dim)``. Set to
        ``None`` when ``use_attention=False``.

    Examples
    --------
    >>> import numpy as np
    >>> from maldideepkit import MaldiMLPClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((64, 256)).astype("float32")
    >>> y = rng.integers(0, 2, size=64)
    >>> clf = MaldiMLPClassifier(epochs=2, batch_size=16, random_state=0).fit(X, y)
    >>> clf.predict(X).shape
    (64,)
    >>> weights = clf.get_attention_weights(X[:4])
    >>> weights.shape
    (4, 512)
    """

    def __init__(
        self,
        input_dim: int | None = None,
        n_classes: int = 2,
        hidden_dim: int = 512,
        head_dims: tuple[int, ...] = (256, 128),
        use_attention: bool = True,
        dropout_high: float = 0.3,
        dropout_low: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip_norm: float | None = None,
        label_smoothing: float = 0.0,
        loss: str = "cross_entropy",
        focal_gamma: float = 2.0,
        use_amp: bool = False,
        swa_start_epoch: int | None = None,
        tune_threshold: bool = False,
        threshold_metric: str = "balanced_accuracy",
        calibrate_temperature: bool = False,
        min_val_auroc_for_threshold_tune: float = 0.6,
        use_sam: bool = False,
        sam_rho: float = 0.05,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        val_fraction: float = 0.1,
        warmup_epochs: int = 0,
        standardize: bool = False,
        input_transform: str | None = None,
        warping: Any | None = None,
        metrics_log_path: str | Path | None = None,
        track_train_metrics: bool = False,
        augment: Any | None = None,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        ema_decay: float | None = None,
        retry_on_val_auroc_below: float | None = None,
        max_retries: int = 2,
        class_weight: str | np.ndarray | list | None = None,
        device: str | torch.device = "auto",
        random_state: int = 0,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            n_classes=n_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            label_smoothing=label_smoothing,
            loss=loss,
            focal_gamma=focal_gamma,
            use_amp=use_amp,
            swa_start_epoch=swa_start_epoch,
            tune_threshold=tune_threshold,
            threshold_metric=threshold_metric,
            calibrate_temperature=calibrate_temperature,
            min_val_auroc_for_threshold_tune=min_val_auroc_for_threshold_tune,
            use_sam=use_sam,
            sam_rho=sam_rho,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            val_fraction=val_fraction,
            warmup_epochs=warmup_epochs,
            standardize=standardize,
            input_transform=input_transform,
            warping=warping,
            metrics_log_path=metrics_log_path,
            track_train_metrics=track_train_metrics,
            augment=augment,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            ema_decay=ema_decay,
            retry_on_val_auroc_below=retry_on_val_auroc_below,
            max_retries=max_retries,
            class_weight=class_weight,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )
        self.hidden_dim = hidden_dim
        self.head_dims = head_dims
        self.use_attention = use_attention
        self.dropout_high = dropout_high
        self.dropout_low = dropout_low
        self.attention_weights_: np.ndarray | None = None

    def _build_model(self) -> nn.Module:
        return SpectralAttentionMLP(
            input_dim=self.input_dim_,
            n_classes=self.n_classes_,
            hidden_dim=int(self.hidden_dim),
            head_dims=tuple(self.head_dims),
            use_attention=bool(self.use_attention),
            dropout_high=float(self.dropout_high),
            dropout_low=float(self.dropout_low),
        )

    def _forward_logits(self, X: Any) -> np.ndarray:
        logits = super()._forward_logits(X)
        if self.use_attention and self.model_.last_attention is not None:
            self.attention_weights_ = self.model_.last_attention.detach().cpu().numpy()
        else:
            self.attention_weights_ = None
        return logits

    def fit(self, X: Any, y: Any) -> MaldiMLPClassifier:  # type: ignore[override]
        """Fit the model and cache attention weights from the final batch.

        See :meth:`BaseSpectralClassifier.fit` for shared parameters.
        """
        super().fit(X, y)
        if self.use_attention:
            X_np = _to_numpy(X)
            tail = X_np[: min(len(X_np), 64)]
            self._forward_logits(tail)
        else:
            self.attention_weights_ = None
        return self

    def get_attention_weights(self, X: Any) -> np.ndarray:
        """Return attention weights for ``X`` of shape ``(len(X), hidden_dim)``.

        Parameters
        ----------
        X : array-like or MaldiSet of shape (n_samples, n_bins)
            Spectra to inspect. Must match ``input_dim_``.

        Returns
        -------
        ndarray of shape (n_samples, hidden_dim)
            Sigmoid-gated attention weights.

        Raises
        ------
        RuntimeError
            If the classifier was built with ``use_attention=False``.
        """
        check_is_fitted(self, "model_")
        if not self.use_attention:
            raise RuntimeError(
                "get_attention_weights is only available when use_attention=True."
            )
        self._forward_logits(X)
        if self.attention_weights_ is None:
            raise RuntimeError(
                "Attention weights were not captured during forward; "
                "ensure the model was built with use_attention=True."
            )
        return self.attention_weights_

    @classmethod
    def from_spectrum(
        cls, bin_width: int, input_dim: int, **overrides
    ) -> "MaldiMLPClassifier":
        """Construct a classifier for a given ``(bin_width, input_dim)`` layout.

        The MLP is architecturally scale-agnostic, so this factory
        only forwards ``input_dim`` and any ``**overrides``. Provided
        for API symmetry with the other classifiers.
        """
        del bin_width
        kwargs: dict[str, Any] = {"input_dim": input_dim}
        kwargs.update(overrides)
        return cls(**kwargs)
