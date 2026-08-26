"""probekit: known-good primitives for linear-probe + steering experiments on HF LMs."""

from .activations import ActivationSet, check_off_by_one, extract, residual_hooks, response_mask
from .data import Example, balance_report, instructed_pairs, load_got, sentiment_small, split_by_group, to_arrays
from .generate import GenOut, generate, logit_diff, next_token_logits, prompting_baseline
from .models import LoadedModel, load_model, pick_device, seed_all, vram_report
from .probes import (LinearProbe, aggregate, cv_auroc, fit_at_layer, layer_sweep, out_of_fold_scores,
                     recall_at_fpr, shuffled_label_null, transfer_matrix)
from .stats import BootResult, auroc_safe, bootstrap_metric, paired_bootstrap_diff, permutation_null
from .steering import ablate, dose_response, random_direction, steer, typical_residual_norm, unit

__all__ = [n for n in dir() if not n.startswith("_")]
