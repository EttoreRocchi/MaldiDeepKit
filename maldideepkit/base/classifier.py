"""Abstract base class for sklearn-compatible spectral classifiers."""

from __future__ import annotations

import json
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader

from ..utils.loss import FocalLoss
from ..utils.reproducibility import resolve_device, seed_everything
from ..utils.training import EarlyStopping, train_loop
from .data import SpectralDataset, _to_numpy, make_loaders


def _serialise_transform_state(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert numpy arrays inside an input-transform state to lists (JSON-safe)."""
    if state is None:
        return None
    out: dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _deserialise_transform_state(
    state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Inverse of :func:`_serialise_transform_state`; numpy-ifies list values."""
    if state is None:
        return None
    out: dict[str, Any] = {}
    for k, v in state.items():
        if k == "mode":
            out[k] = v
        elif isinstance(v, list):
            out[k] = np.asarray(v, dtype=np.float32)
        else:
            out[k] = v
    return out


class BaseSpectralClassifier(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):  # type: ignore[misc]
    """Abstract base for all MaldiDeepKit classifiers.

    Concrete subclasses only need to override :meth:`_build_model`,
    which should return a :class:`torch.nn.Module` that maps an input of
    shape ``(batch, input_dim)`` to logits of shape ``(batch, n_classes)``.
    Everything else (device placement, validation split, early stopping,
    checkpointing, predict / predict_proba, save / load) is provided here.

    Parameters
    ----------
    input_dim : int or None, default=None
        Number of input bins. If ``None``, inferred from ``X`` at
        :meth:`fit` time and stored as :attr:`input_dim_`.
    n_classes : int, default=2
        Number of output classes. Overwritten with the true number of
        classes found in ``y`` at :meth:`fit` time.
    learning_rate : float, default=1e-3
        Initial learning rate for the optimizer (Adam by default; AdamW
        when ``weight_decay > 0``).
    weight_decay : float, default=0.0
        L2 penalty applied via decoupled weight decay. When ``> 0`` the
        optimizer switches from ``Adam`` to ``AdamW``.
    grad_clip_norm : float or None, default=None
        If set, clip gradient global L2 norm to this value before every
        optimizer step. ``1.0`` is a common default for transformers.
    label_smoothing : float, default=0.0
        Label smoothing factor in ``[0, 1)`` passed to the loss. Applied
        to both cross-entropy and focal-loss paths.
    loss : {"cross_entropy", "focal"}, default="cross_entropy"
        Classification loss. ``"focal"`` uses
        :class:`~maldideepkit.utils.FocalLoss` with ``gamma=focal_gamma``.
        Good for highly imbalanced problems.
    focal_gamma : float, default=2.0
        Focal-loss focusing parameter. Ignored when
        ``loss="cross_entropy"``.
    use_amp : bool, default=False
        If ``True`` and the resolved device is CUDA, run forward + loss
        under :func:`torch.autocast` and use :class:`torch.amp.GradScaler`
        for backward. ~2x wall-time speedup on recent NVIDIA GPUs. On CPU
        this is a no-op.
    swa_start_epoch : int or None, default=None
        If set, start Stochastic Weight Averaging at this epoch. The SWA
        average replaces the best-val checkpoint at the end of fit.
        Typical value: 60-80% of ``epochs``.
    tune_threshold : bool, default=False
        (Binary classification only.) After fit, sweep thresholds on
        the validation split and store the one that maximises
        ``threshold_metric``. :meth:`predict` uses this threshold
        instead of ``argmax @ 0.5``.
    threshold_metric : {"balanced_accuracy", "f1", "youden"}, default="balanced_accuracy"
        Metric used by ``tune_threshold``.
    calibrate_temperature : bool, default=False
        If ``True``, after fit run LBFGS-based temperature scaling on
        held-out validation logits (Guo et al. 2017). The fitted
        temperature is stored as :attr:`temperature_` and applied in
        :meth:`predict_proba` to sharpen / smooth probabilities
        without changing the argmax.
    min_val_auroc_for_threshold_tune : float, default=0.6
        Binary-classification guardrail on ``tune_threshold=True``: if
        the validation AUROC is below this value, the threshold sweep
        is skipped and ``threshold_`` falls back to ``0.5``. Set to
        ``0.0`` to disable.
    use_sam : bool, default=False
        If ``True``, wrap the base optimizer in
        :class:`~maldideepkit.utils.SAMOptimizer` and run the two-step
        Sharpness-Aware Minimization update. Doubles forward / backward
        compute per step; typically helps generalisation on small
        datasets.
    sam_rho : float, default=0.05
        Size of the SAM ascent step. Ignored when ``use_sam=False``.
    batch_size : int, default=32
        Training mini-batch size.
    epochs : int, default=100
        Maximum number of training epochs.
    early_stopping_patience : int, default=10
        Number of epochs without validation-loss improvement before
        training is stopped.
    val_fraction : float, default=0.1
        Fraction of the training data held out for the internal
        validation split.
    warmup_epochs : int, default=0
        If positive, linearly ramp each optimizer param group's learning
        rate from ``0`` to its configured target over the first
        ``warmup_epochs`` epochs. Useful for transformer architectures
        that can diverge at full learning rate during the first few steps.
    standardize : bool, default=False
        Shorthand for ``input_transform="standardize"`` (when True) or
        ``"none"`` (when False). Kept for backwards compatibility;
        ``input_transform`` is the modern interface and wins when
        both are supplied.
    input_transform : str, optional
        One of ``{"none", "standardize", "log1p", "robust",
        "log1p+standardize"}``. Fit on the (warped) training split
        only and stored as :attr:`input_transform_state_`; reapplied
        at :meth:`predict` / :meth:`predict_proba` time.
    warping : sklearn-style transformer, optional
        Spectral alignment / warping transformer applied **before**
        standardization. Fitted on the training split only, then
        used to transform both splits during training and new data
        at :meth:`predict` / :meth:`predict_proba` time. The fitted
        transformer is stored as :attr:`warper_`.
    metrics_log_path : str or Path, optional
        If set, write a per-epoch metrics CSV to this path during
        :meth:`fit`. One row per epoch with columns ``epoch,
        train_loss, val_loss, lr, mean_grad_norm, n_grad_updates``
        (+ ``train_auroc, val_auroc`` when ``track_train_metrics=True``).
    track_train_metrics : bool, default=False
        Only used when ``metrics_log_path`` is set. If ``True``, after
        every epoch run a no-grad forward pass over the full training
        split and record ``train_auroc`` + ``val_auroc`` alongside the
        losses. Adds one extra pass per epoch; binary classification
        only.
    augment : callable, optional
        Per-batch augmentation applied to training batches only. The
        usual choice is :class:`~maldideepkit.augment.SpectrumAugment`.
    mixup_alpha : float, default=0.0
        If positive, apply MixUp augmentation per training batch with a
        Beta(``mixup_alpha``, ``mixup_alpha``) mixing coefficient.
        ``0.0`` disables MixUp. Composable with ``cutmix_alpha``.
    cutmix_alpha : float, default=0.0
        If positive, apply CutMix augmentation per training batch with
        a Beta(``cutmix_alpha``, ``cutmix_alpha``) mixing coefficient.
        ``0.0`` disables CutMix.
    ema_decay : float or None, default=None
        If set (typically ``0.999``), maintain an exponential moving
        average of model weights during training and use the EMA weights
        at inference time.
    retry_on_val_auroc_below : float or None, default=None
        Binary-classification guardrail. If set and the post-fit
        validation AUROC is below this threshold, retrain with a
        different RNG seed up to ``max_retries`` times. Useful for
        unstable small-data fits.
    max_retries : int, default=2
        Maximum number of automatic refits triggered by
        ``retry_on_val_auroc_below``. Ignored when that guardrail is
        unset.
    class_weight : {"balanced", None} or array-like, default=None
        Per-class weights applied to :class:`~torch.nn.CrossEntropyLoss`.
        ``"balanced"`` uses ``n_samples / (n_classes * class_count)``.
    device : {"auto", "cpu", "cuda"} or torch.device, default="auto"
        Device used for training and inference.
    random_state : int, default=0
        Seeds Python, NumPy, and PyTorch RNGs and the validation split.
    verbose : bool, default=False
        If ``True``, prints one line per training epoch.

    Attributes
    ----------
    model_ : torch.nn.Module
        The fitted PyTorch model.
    classes_ : ndarray of shape (n_classes,)
        Original class labels seen during :meth:`fit`.
    input_dim_ : int
        Resolved number of input features.
    n_classes_ : int
        Resolved number of classes.
    feature_mean_ : ndarray or None
        Per-feature mean used when ``standardize=True``.
    feature_std_ : ndarray or None
        Per-feature std used when ``standardize=True``.
    n_features_in_ : int
        Number of features seen at :meth:`fit` (sklearn convention).
    """

    def __init__(
        self,
        input_dim: int | None = None,
        n_classes: int = 2,
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
        augment: Callable[[torch.Tensor], torch.Tensor] | None = None,
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
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.label_smoothing = label_smoothing
        self.loss = loss
        self.focal_gamma = focal_gamma
        self.use_amp = use_amp
        self.swa_start_epoch = swa_start_epoch
        self.tune_threshold = tune_threshold
        self.threshold_metric = threshold_metric
        self.calibrate_temperature = calibrate_temperature
        self.min_val_auroc_for_threshold_tune = min_val_auroc_for_threshold_tune
        self.use_sam = use_sam
        self.sam_rho = sam_rho
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.val_fraction = val_fraction
        self.warmup_epochs = warmup_epochs
        self.standardize = standardize
        self.input_transform = input_transform
        self.warping = warping
        self.metrics_log_path = metrics_log_path
        self.track_train_metrics = track_train_metrics
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.ema_decay = ema_decay
        self.retry_on_val_auroc_below = retry_on_val_auroc_below
        self.max_retries = max_retries
        self.class_weight = class_weight
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Return a fresh :class:`nn.Module` for the current hyperparameters.

        Implementations should use :attr:`input_dim_` and
        :attr:`n_classes_` rather than the constructor arguments, since
        those are the values resolved at :meth:`fit` time.
        """

    def _optimizer_param_groups(self, model: nn.Module) -> list[dict[str, Any]]:
        """Return parameter groups for the optimizer.

        Default: a single group containing every parameter of ``model``.
        Override in a subclass to give specific parameters a different
        learning rate or weight decay.

        Parameters
        ----------
        model : nn.Module
            The freshly built, on-device training model.

        Returns
        -------
        list of dict
            Param-group dicts suitable for :class:`torch.optim.Adam`.
            Groups without an explicit ``lr`` or ``weight_decay``
            inherit the defaults from :meth:`fit`.
        """
        return [{"params": list(model.parameters())}]

    def _resolve_device(self) -> torch.device:
        return resolve_device(self.device)

    def _compute_class_weight(self, y: np.ndarray) -> torch.Tensor | None:
        if self.class_weight is None:
            return None
        if isinstance(self.class_weight, str):
            if self.class_weight != "balanced":
                raise ValueError(
                    f"Unknown class_weight={self.class_weight!r}; "
                    "use 'balanced', None, or an array."
                )
            counts = np.bincount(y, minlength=self.n_classes_)
            if np.any(counts == 0):
                missing = np.flatnonzero(counts == 0).tolist()
                raise ValueError(
                    "class_weight='balanced' requires every class to be "
                    f"present in y; missing class indices: {missing}."
                )
            weights = len(y) / (self.n_classes_ * counts)
            return torch.tensor(weights, dtype=torch.float32)
        weights = np.asarray(self.class_weight, dtype=np.float32)
        if weights.shape != (self.n_classes_,):
            raise ValueError(
                f"class_weight has shape {weights.shape}, "
                f"expected ({self.n_classes_},)."
            )
        return torch.from_numpy(weights)

    def _build_criterion(self, class_weight: torch.Tensor | None) -> nn.Module:
        if self.loss == "cross_entropy":
            return nn.CrossEntropyLoss(
                weight=class_weight,
                label_smoothing=float(self.label_smoothing),
            )
        if self.loss == "focal":
            return FocalLoss(
                weight=class_weight,
                gamma=float(self.focal_gamma),
                label_smoothing=float(self.label_smoothing),
            )
        raise ValueError(
            f"Unknown loss={self.loss!r}; expected 'cross_entropy' or 'focal'."
        )

    def _make_mix_generator(self) -> torch.Generator | None:
        if float(self.mixup_alpha) <= 0 and float(self.cutmix_alpha) <= 0:
            return None
        gen = torch.Generator()
        gen.manual_seed(int(self.random_state))
        return gen

    def _prepare_inputs(self, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        X_np = _to_numpy(X)
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()
        y_np = np.asarray(y).ravel()
        if X_np.shape[0] != y_np.shape[0]:
            raise ValueError(f"X has {X_np.shape[0]} rows but y has {y_np.shape[0]}.")
        self.classes_ = unique_labels(y_np)
        self.n_classes_ = int(len(self.classes_))
        if self.n_classes_ < 2:
            raise ValueError(
                f"{type(self).__name__} needs at least 2 classes in y; "
                f"got {self.n_classes_}."
            )
        self.input_dim_ = (
            int(X_np.shape[1]) if self.input_dim is None else int(self.input_dim)
        )
        if self.input_dim_ != X_np.shape[1]:
            raise ValueError(
                f"input_dim={self.input_dim_} does not match X.shape[1]={X_np.shape[1]}."
            )
        self.n_features_in_ = self.input_dim_
        y_encoded = np.searchsorted(self.classes_, y_np).astype(np.int64)
        return X_np, y_encoded

    def fit(self, X: Any, y: Any) -> BaseSpectralClassifier:
        """Fit the model on ``(X, y)``.

        Parameters
        ----------
        X : array-like or MaldiSet of shape (n_samples, n_bins)
            Training spectra. NumPy arrays, pandas DataFrames, and
            objects with a DataFrame-like ``.X`` attribute are accepted.
        y : array-like of shape (n_samples,)
            Integer or string class labels. Re-encoded to ``0..n_classes-1``
            internally; original labels are preserved in :attr:`classes_`.

        Returns
        -------
        self : BaseSpectralClassifier
            The fitted estimator.
        """
        seed_everything(int(self.random_state))
        X_np, y_encoded = self._prepare_inputs(X, y)
        device = self._resolve_device()

        warper = clone(self.warping) if self.warping is not None else None
        train_loader, val_loader, stats = make_loaders(
            X_np,
            y_encoded,
            batch_size=int(self.batch_size),
            val_size=float(self.val_fraction),
            random_state=int(self.random_state),
            standardize=bool(self.standardize),
            input_transform=self.input_transform,
            warper=warper,
        )
        self.feature_mean_ = stats["mean"]
        self.feature_std_ = stats["std"]
        self.warper_ = stats["warper"]
        self.input_transform_state_ = stats["input_transform_state"]

        class_weight = self._compute_class_weight(y_encoded)
        if class_weight is not None:
            class_weight = class_weight.to(device)
        criterion = self._build_criterion(class_weight)

        X_val_t, y_val_t = self._collect_validation(val_loader, device)

        # Retry-on-collapse: when ``retry_on_val_auroc_below`` is set,
        # reseed and retrain up to ``max_retries`` more times if the
        # final val AUROC is below threshold.
        retry_threshold = self.retry_on_val_auroc_below
        max_retries = max(0, int(self.max_retries))
        total_attempts = 1 + (max_retries if retry_threshold is not None else 0)
        final_val_auroc = float("nan")
        base_seed = int(self.random_state)
        fitted_model: nn.Module | None = None
        for attempt in range(total_attempts):
            if attempt > 0:
                seed_everything(base_seed + 1_000_003 * attempt)

            model = self._build_model().to(device)

            opt_cls = (
                torch.optim.AdamW if float(self.weight_decay) > 0 else torch.optim.Adam
            )
            param_groups = self._optimizer_param_groups(model)
            if bool(self.use_sam):
                from ..utils.sam import SAMOptimizer

                optimizer = SAMOptimizer(
                    param_groups,
                    base_optimizer=opt_cls,
                    rho=float(self.sam_rho),
                    lr=float(self.learning_rate),
                    weight_decay=float(self.weight_decay),
                )
            else:
                optimizer = opt_cls(
                    param_groups,
                    lr=float(self.learning_rate),
                    weight_decay=float(self.weight_decay),
                )
            warmup = max(0, int(self.warmup_epochs))
            t_max = max(1, int(self.epochs) - warmup)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=1e-6
            )

            metrics_recorder = self._build_metrics_recorder(
                model, train_loader, X_val_t, y_val_t, device
            )

            early = EarlyStopping(patience=int(self.early_stopping_patience))
            fitted_model = train_loop(
                model,
                train_loader,
                (X_val_t, y_val_t),
                criterion,
                optimizer,
                scheduler,
                device,
                int(self.epochs),
                early,
                verbose=bool(self.verbose),
                warmup_epochs=int(self.warmup_epochs),
                grad_clip_norm=(
                    float(self.grad_clip_norm)
                    if self.grad_clip_norm is not None
                    else None
                ),
                use_amp=bool(self.use_amp),
                swa_start_epoch=(
                    int(self.swa_start_epoch)
                    if self.swa_start_epoch is not None
                    else None
                ),
                use_sam=bool(self.use_sam),
                metrics_recorder=metrics_recorder,
                augment=self.augment,
                mixup_alpha=float(self.mixup_alpha),
                cutmix_alpha=float(self.cutmix_alpha),
                n_classes=int(self.n_classes_),
                mix_generator=self._make_mix_generator(),
                ema_decay=(
                    float(self.ema_decay) if self.ema_decay is not None else None
                ),
            )

            if retry_threshold is None or attempt == total_attempts - 1:
                break
            final_val_auroc = self._attempt_val_auroc(
                fitted_model, X_val_t, y_val_t, device
            )
            if not np.isfinite(final_val_auroc) or final_val_auroc >= float(
                retry_threshold
            ):
                break
        assert fitted_model is not None
        self.model_ = fitted_model
        self._device_ = device
        self.threshold_ = None
        self.temperature_ = None

        if self.metrics_log_path is not None:
            self._write_post_fit_sidecar(X_val_t, y_val_t)

        if bool(self.tune_threshold) or bool(self.calibrate_temperature):
            self._fit_post_hoc_calibration(X_val_t, y_val_t)

        return self

    def _attempt_val_auroc(
        self,
        model: nn.Module,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        device: torch.device,
    ) -> float:
        """Return binary val AUROC of a freshly-fitted model, or NaN.

        Returns ``NaN`` for multi-class or when the val split has only
        one class present.
        """
        from sklearn.metrics import roc_auc_score

        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        y_np = y_val.detach().cpu().numpy()
        if probs.shape[1] != 2 or len(np.unique(y_np)) < 2:
            return float("nan")
        try:
            return float(roc_auc_score(y_np, probs[:, 1]))
        except ValueError:
            return float("nan")

    def _write_post_fit_sidecar(self, X_val: torch.Tensor, y_val: torch.Tensor) -> None:
        """Compute val loss + AUROC on the deployed model and write a JSON sidecar."""
        from sklearn.metrics import roc_auc_score

        log_path = Path(self.metrics_log_path)
        sidecar_path = log_path.with_suffix(log_path.suffix + ".post_fit.json")

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_val)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            y_np = y_val.detach().cpu().numpy()
            ce = nn.CrossEntropyLoss()(logits, y_val)
            val_loss = float(ce.item())

        val_auroc: float | None = None
        n_classes = probs.shape[1]
        try:
            if n_classes == 2:
                val_auroc = float(roc_auc_score(y_np, probs[:, 1]))
            else:
                val_auroc = float(
                    roc_auc_score(y_np, probs, multi_class="ovr", average="macro")
                )
        except ValueError:
            pass

        if self.ema_decay is not None:
            source = "ema"
        elif self.swa_start_epoch is not None:
            source = "swa"
        else:
            source = "best_val"

        payload = {
            "val_loss": val_loss,
            "val_auroc": val_auroc,
            "weights_source": source,
            "n_classes": int(n_classes),
        }
        sidecar_path.write_text(json.dumps(payload, indent=2))

    def _build_metrics_recorder(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        X_val_t: torch.Tensor,
        y_val_t: torch.Tensor,
        device: torch.device,
    ) -> Callable[[dict[str, float]], None] | None:
        """Return a per-epoch recorder that appends diagnostics to a CSV.

        Returns ``None`` (no recording) when ``metrics_log_path`` is unset.
        """
        if self.metrics_log_path is None:
            return None

        log_path = Path(self.metrics_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            log_path.unlink()

        track_train = bool(self.track_train_metrics)
        binary = self.n_classes_ == 2

        def _collect_train_val_auroc() -> tuple[float, float]:
            from sklearn.metrics import roc_auc_score

            if not binary:
                import math

                return math.nan, math.nan
            y_val_np = y_val_t.detach().cpu().numpy()
            y_tr_parts: list[np.ndarray] = []
            proba_tr_parts: list[np.ndarray] = []
            model.eval()
            with torch.no_grad():
                for xb, yb in train_loader:
                    xb = xb.to(device, non_blocking=True)
                    logits = model(xb).detach()
                    proba = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                    proba_tr_parts.append(proba)
                    y_tr_parts.append(np.asarray(yb).ravel())
                val_logits = model(X_val_t).detach()
                val_proba = torch.softmax(val_logits, dim=-1)[:, 1].cpu().numpy()
            model.train()
            y_tr = np.concatenate(y_tr_parts)
            proba_tr = np.concatenate(proba_tr_parts)
            try:
                train_auroc = float(roc_auc_score(y_tr, proba_tr))
            except ValueError:
                train_auroc = float("nan")
            try:
                val_auroc = float(roc_auc_score(y_val_np, val_proba))
            except ValueError:
                val_auroc = float("nan")
            return train_auroc, val_auroc

        header_written = False

        def recorder(payload: dict[str, float]) -> None:
            nonlocal header_written
            row: dict[str, float | int] = dict(payload)
            if track_train:
                train_auroc, val_auroc = _collect_train_val_auroc()
                row["train_auroc"] = train_auroc
                row["val_auroc"] = val_auroc
            columns = [
                "epoch",
                "train_loss",
                "val_loss",
                "lr",
                "mean_grad_norm",
                "n_grad_updates",
            ]
            if track_train:
                columns += ["train_auroc", "val_auroc"]
            with open(log_path, "a") as fh:
                if not header_written:
                    fh.write(",".join(columns) + "\n")
                    header_written = True
                fh.write(",".join(str(row.get(k, "")) for k in columns) + "\n")

        return recorder

    def _fit_post_hoc_calibration(
        self, X_val_t: torch.Tensor, y_val_t: torch.Tensor
    ) -> None:
        """Collect held-out logits once, then fit threshold / temperature."""
        from ..utils.calibration import fit_temperature, tune_threshold

        self.model_.eval()
        with torch.no_grad():
            val_logits = self.model_(X_val_t).detach().cpu()
        y_val_np = y_val_t.detach().cpu().numpy()

        if bool(self.calibrate_temperature):
            self.temperature_ = float(fit_temperature(val_logits, y_val_np))

        if bool(self.tune_threshold):
            if self.n_classes_ != 2:
                self.threshold_ = None
            else:
                from sklearn.metrics import roc_auc_score

                logits_np = val_logits.numpy()
                if self.temperature_ is not None:
                    logits_np = logits_np / float(self.temperature_)
                logits_np = logits_np - logits_np.max(axis=1, keepdims=True)
                exp = np.exp(logits_np)
                proba = exp / exp.sum(axis=1, keepdims=True)
                try:
                    val_auroc = float(roc_auc_score(y_val_np, proba[:, 1]))
                except ValueError:
                    val_auroc = float("nan")
                gate = float(self.min_val_auroc_for_threshold_tune)
                if np.isfinite(val_auroc) and val_auroc >= gate:
                    self.threshold_ = float(
                        tune_threshold(
                            y_val_np, proba[:, 1], metric=self.threshold_metric
                        )
                    )
                else:
                    import logging

                    logging.getLogger(__name__).info(
                        "tune_threshold skipped: val AUROC=%.3f < %.2f; "
                        "falling back to threshold_=0.5",
                        val_auroc,
                        gate,
                    )
                    self.threshold_ = 0.5

    @staticmethod
    def _collect_validation(
        val_loader: DataLoader, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for xb, yb in val_loader:
            xs.append(xb)
            ys.append(yb)
        X_val = torch.cat(xs, dim=0).to(device)
        y_val = torch.cat(ys, dim=0).to(device)
        return X_val, y_val

    def _check_input_dim(self, X: np.ndarray) -> None:
        if X.shape[1] != self.input_dim_:
            raise ValueError(
                f"X has {X.shape[1]} features but estimator was fitted with "
                f"input_dim={self.input_dim_}. Retrain or re-bin your data "
                "to match the original resolution."
            )

    def _forward_logits(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "model_")
        X_np = _to_numpy(X)
        self._check_input_dim(X_np)
        if getattr(self, "warper_", None) is not None:
            from .data import _warp_numpy

            X_np = _warp_numpy(self.warper_, X_np)
        state = getattr(self, "input_transform_state_", None)
        if state is not None and state.get("mode", "none") != "none":
            from .data import apply_input_transform

            X_np = apply_input_transform(X_np, state)
        elif self.standardize and self.feature_mean_ is not None:
            from .data import _STD_FLOOR

            safe_std = np.maximum(self.feature_std_, _STD_FLOOR).astype(np.float32)
            X_np = (X_np - self.feature_mean_) / safe_std
        device = self._device_
        self.model_.eval()
        # Batch inference so large test folds don't OOM on attention
        # architectures.
        X_t_full = torch.from_numpy(X_np.astype(np.float32)).to(device)
        chunk = int(getattr(self, "batch_size", 32))
        chunk = max(1, chunk)
        logits_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, X_t_full.shape[0], chunk):
                X_t = X_t_full[start : start + chunk]
                logits_chunks.append(self.model_(X_t).detach().cpu().numpy())
        logits = (
            np.concatenate(logits_chunks, axis=0)
            if logits_chunks
            else np.empty((0, self.n_classes_), dtype=np.float32)
        )
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)
        return logits

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return softmax class probabilities of shape ``(n_samples, n_classes)``.

        Parameters
        ----------
        X : array-like or MaldiSet of shape (n_samples, n_bins)
            Spectra to score. Must have the same number of features as
            the training matrix.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Softmax probabilities that sum to 1 along the class axis.

        Raises
        ------
        ValueError
            If ``X.shape[1] != input_dim_``.
        """
        logits = self._forward_logits(X)
        temperature = getattr(self, "temperature_", None)
        if temperature is not None:
            logits = logits / float(temperature)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        """Return hard class predictions.

        Parameters
        ----------
        X : array-like or MaldiSet of shape (n_samples, n_bins)
            Spectra to classify.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted labels, drawn from :attr:`classes_`.

        Notes
        -----
        For binary classifiers fit with ``tune_threshold=True``, the
        decision uses the fitted :attr:`threshold_` on the positive
        class probability instead of ``argmax``.
        """
        proba = self.predict_proba(X)
        threshold = getattr(self, "threshold_", None)
        if threshold is not None and proba.shape[1] == 2:
            idx = (proba[:, 1] >= float(threshold)).astype(int)
        else:
            idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def score(self, X: Any, y: Any) -> float:
        """Return mean accuracy on ``(X, y)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_bins)
        y : array-like of shape (n_samples,)

        Returns
        -------
        float
            Accuracy in ``[0, 1]``.
        """
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()
        y = np.asarray(y).ravel()
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def _hparam_dict(self) -> dict[str, Any]:
        params = self.get_params(deep=False)
        if isinstance(params.get("device"), torch.device):
            params["device"] = str(params["device"])
        if isinstance(params.get("class_weight"), np.ndarray):
            params["class_weight"] = params["class_weight"].tolist()
        params["warping"] = None if params.get("warping") is None else "<provided>"
        return params

    def save(self, path: str | Path) -> None:
        """Persist the fitted estimator to ``path.pt`` + ``path.json``.

        The PyTorch state dict is written to ``<path>.pt`` and the
        hyperparameters plus fitted metadata to ``<path>.json``. A
        single ``.pt`` or ``.json`` suffix on ``path`` is stripped
        so ``clf.save("model")`` and ``clf.save("model.pt")`` produce
        the same pair of files.

        Parameters
        ----------
        path : str or Path
            Base path without extension.
        """
        check_is_fitted(self, "model_")
        base = Path(path)
        if base.suffix in {".pt", ".pth", ".json"}:
            base = base.with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.model_.state_dict(), base.with_suffix(".pt"))
        warper = getattr(self, "warper_", None)
        warper_path = base.with_suffix(".warper.pkl")
        if warper is not None:
            import joblib

            joblib.dump(warper, warper_path)
        elif warper_path.exists():
            warper_path.unlink()

        meta: dict[str, Any] = {
            "class_name": type(self).__name__,
            "version": 2,
            "hparams": self._hparam_dict(),
            "fitted": {
                "input_dim_": int(self.input_dim_),
                "n_classes_": int(self.n_classes_),
                "classes_": np.asarray(self.classes_).tolist(),
                "n_features_in_": int(self.n_features_in_),
                "feature_mean_": (
                    None
                    if self.feature_mean_ is None
                    else np.asarray(self.feature_mean_).tolist()
                ),
                "feature_std_": (
                    None
                    if self.feature_std_ is None
                    else np.asarray(self.feature_std_).tolist()
                ),
                "threshold_": getattr(self, "threshold_", None),
                "temperature_": getattr(self, "temperature_", None),
                "has_warper": warper is not None,
                "input_transform_state_": _serialise_transform_state(
                    getattr(self, "input_transform_state_", None)
                ),
            },
        }
        with open(base.with_suffix(".json"), "w") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> BaseSpectralClassifier:
        """Load a saved estimator from a ``save()``-produced pair of files.

        Parameters
        ----------
        path : str or Path
            Base path (``.pt``/``.json`` suffix optional).

        Returns
        -------
        BaseSpectralClassifier
            Fitted estimator ready for :meth:`predict` /
            :meth:`predict_proba`.

        Raises
        ------
        ValueError
            If the JSON file identifies a different class from ``cls``.
        FileNotFoundError
            If either ``.pt`` or ``.json`` is missing.
        """
        base = Path(path)
        if base.suffix in {".pt", ".pth", ".json"}:
            base = base.with_suffix("")
        pt_path = base.with_suffix(".pt")
        json_path = base.with_suffix(".json")
        if not pt_path.exists():
            raise FileNotFoundError(pt_path)
        if not json_path.exists():
            raise FileNotFoundError(json_path)

        with open(json_path) as fh:
            meta = json.load(fh)

        if cls is not BaseSpectralClassifier and meta["class_name"] != cls.__name__:
            raise ValueError(
                f"Saved class is {meta['class_name']!r} but load() was called "
                f"on {cls.__name__!r}."
            )

        target_cls = cls
        if cls is BaseSpectralClassifier:
            from .. import (
                MaldiCNNClassifier,
                MaldiMLPClassifier,
                MaldiResNetClassifier,
                MaldiTransformerClassifier,
            )

            registry = {
                c.__name__: c
                for c in (
                    MaldiMLPClassifier,
                    MaldiCNNClassifier,
                    MaldiResNetClassifier,
                    MaldiTransformerClassifier,
                )
            }
            if meta["class_name"] not in registry:
                raise ValueError(f"Unknown saved class: {meta['class_name']!r}")
            target_cls = registry[meta["class_name"]]

        hparams = dict(meta["hparams"])
        hparams.pop("warping", None)
        instance = target_cls(**hparams)
        fitted = meta["fitted"]
        instance.input_dim_ = int(fitted["input_dim_"])
        instance.n_classes_ = int(fitted["n_classes_"])
        instance.classes_ = np.asarray(fitted["classes_"])
        instance.n_features_in_ = int(fitted["n_features_in_"])
        instance.feature_mean_ = (
            None
            if fitted["feature_mean_"] is None
            else np.asarray(fitted["feature_mean_"], dtype=np.float32)
        )
        instance.feature_std_ = (
            None
            if fitted["feature_std_"] is None
            else np.asarray(fitted["feature_std_"], dtype=np.float32)
        )
        instance.threshold_ = fitted.get("threshold_")
        instance.temperature_ = fitted.get("temperature_")
        instance.input_transform_state_ = _deserialise_transform_state(
            fitted.get("input_transform_state_")
        )
        instance.warper_ = None
        if fitted.get("has_warper"):
            import joblib

            warper_path = base.with_suffix(".warper.pkl")
            if not warper_path.exists():
                raise FileNotFoundError(warper_path)
            instance.warper_ = joblib.load(warper_path)

        device = resolve_device(instance.device)
        model = instance._build_model().to(device)
        state = torch.load(pt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        instance.model_ = model
        instance._device_ = device
        return instance

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "model_")

    def __sklearn_tags__(self):  # pragma: no cover - sklearn >=1.6 tag plumbing
        try:
            tags = super().__sklearn_tags__()
        except AttributeError:
            return None
        tags.input_tags.two_d_array = True
        tags.input_tags.sparse = False
        tags.classifier_tags.multi_class = True
        tags.classifier_tags.poor_score = True
        tags.non_deterministic = False
        return tags

    def _more_tags(self) -> dict[str, Any]:  # pragma: no cover - sklearn <1.6
        return {
            "binary_only": False,
            "multioutput": False,
            "poor_score": True,
            "requires_positive_X": False,
            "X_types": ["2darray"],
        }


__all__ = ["BaseSpectralClassifier", "SpectralDataset"]
