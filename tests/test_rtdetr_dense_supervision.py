# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for RT-DETRv3-style dense positive supervision in the RT-DETR auxiliary loss."""

import torch

from ultralytics.models.utils.dense_assigner import DensePositiveMatcher
from ultralytics.models.utils.loss import RTDETRDetectionLoss


def _toy_inputs(num_layers=3, bs=1, nq=20, nc=4, n_gt=3):
    """Build small synthetic RT-DETR predictions and a ground-truth batch."""
    pred_bboxes = torch.rand(num_layers, bs, nq, 4)  # xywh in [0, 1]
    pred_scores = torch.randn(num_layers, bs, nq, nc)
    batch = {
        "cls": torch.randint(0, nc, (n_gt,)),
        "bboxes": torch.rand(n_gt, 4),
        "batch_idx": torch.zeros(n_gt, dtype=torch.long),
        "gt_groups": [n_gt],
    }
    return (pred_bboxes, pred_scores), batch


def test_dense_matcher_assigns_topk_per_gt():
    """Each ground truth should receive exactly top-k query matches (one-to-many)."""
    topk, n_gt = 4, 3
    matcher = DensePositiveMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2}, topk=topk)
    pred_bboxes = torch.rand(1, 30, 4)
    pred_scores = torch.randn(1, 30, 5)
    gt_bboxes = torch.rand(n_gt, 4)
    gt_cls = torch.randint(0, 5, (n_gt,))

    (query_idx, gt_idx) = matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups=[n_gt])[0]

    assert query_idx.numel() == topk * n_gt
    assert query_idx.max() < 30  # valid query indices
    # Every ground truth index appears exactly topk times.
    counts = torch.bincount(gt_idx, minlength=n_gt)
    assert torch.equal(counts, torch.full((n_gt,), topk))


def test_dense_aux_is_wired_into_rtdetr_loss():
    """RTDETRDetectionLoss(dense_aux=True) wires a DensePositiveMatcher into the auxiliary path."""
    loss_fn = RTDETRDetectionLoss(nc=4, use_vfl=True, dense_aux=True, dense_aux_topk=4)
    assert loss_fn.dense_aux is True
    assert isinstance(loss_fn.dense_matcher, DensePositiveMatcher)

    baseline = RTDETRDetectionLoss(nc=4, use_vfl=True)
    assert baseline.dense_aux is False
    assert baseline.dense_matcher is None


def test_dense_aux_produces_finite_losses():
    """Running the RT-DETR loss with dense auxiliary supervision yields finite loss terms."""
    preds, batch = _toy_inputs()
    loss_fn = RTDETRDetectionLoss(nc=4, use_vfl=True, dense_aux=True, dense_aux_topk=4)

    losses = loss_fn(preds, batch)

    assert "loss_bbox_aux" in losses and "loss_class_aux" in losses
    for v in losses.values():
        assert torch.isfinite(v)


def test_dense_matcher_yields_more_positives_than_hungarian():
    """For identical inputs the dense matcher assigns strictly more positives than one-to-one Hungarian matching."""
    n_gt, topk = 3, 4
    pred_bboxes = torch.rand(1, 30, 4)
    pred_scores = torch.randn(1, 30, 5)
    gt_bboxes = torch.rand(n_gt, 4)
    gt_cls = torch.randint(0, 5, (n_gt,))

    # The base Hungarian matcher is reachable through the loss object's existing `matcher` attribute.
    loss_fn = RTDETRDetectionLoss(nc=5, use_vfl=True, dense_aux=True, dense_aux_topk=topk)
    hungarian = loss_fn.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups=[n_gt])[0]
    dense = loss_fn.dense_matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups=[n_gt])[0]

    assert hungarian[0].numel() == n_gt  # one-to-one: one query per gt
    assert dense[0].numel() == topk * n_gt  # one-to-many: topk queries per gt
    assert dense[0].numel() > hungarian[0].numel()
