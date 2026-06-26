# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Dense positive label assignment for DETR-style auxiliary supervision.

Hungarian matching assigns exactly one query to each ground truth, which gives much sparser supervision than the
one-to-many assignment used by dense detectors such as the YOLO series. This module provides ``DensePositiveMatcher``,
a one-to-many assigner that selects the top-k lowest-cost queries per ground truth using the *same* matching cost as
``HungarianMatcher``. Applying it only to the intermediate (auxiliary) decoder layers densifies the training gradient
while leaving the final one-to-one layer — and therefore NMS-free inference — untouched.

Adapted from the hierarchical dense positive supervision idea in RT-DETRv3 (https://arxiv.org/abs/2409.08475).
"""

from __future__ import annotations

import torch

from .ops import HungarianMatcher


class DensePositiveMatcher(HungarianMatcher):
    """One-to-many assigner that matches each ground truth to its top-k lowest-cost queries.

    Reuses ``HungarianMatcher.build_cost_matrix`` so the per-pair matching cost is identical to the one-to-one matcher;
    only the selection step differs. Instead of a single optimal assignment per ground truth, every ground truth keeps
    its ``topk`` best-scoring queries, producing a denser set of positive samples for auxiliary supervision.

    Attributes:
        topk (int): Number of positive queries assigned to each ground truth.

    Examples:
        >>> matcher = DensePositiveMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2}, topk=4)
        >>> pred_boxes = torch.rand(2, 100, 4)
        >>> pred_scores = torch.rand(2, 100, 80)
        >>> gt_boxes = torch.rand(10, 4)
        >>> gt_classes = torch.randint(0, 80, (10,))
        >>> indices = matcher(pred_boxes, pred_scores, gt_boxes, gt_classes, gt_groups=[5, 5])
    """

    def __init__(self, *args, topk: int = 4, **kwargs):
        """Initialize the dense matcher.

        Args:
            *args (Any): Positional arguments forwarded to ``HungarianMatcher``.
            topk (int): Number of positive queries to assign to each ground truth.
            **kwargs (Any): Keyword arguments forwarded to ``HungarianMatcher``.
        """
        super().__init__(*args, **kwargs)
        self.topk = topk

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        masks: torch.Tensor | None = None,
        gt_mask: list[torch.Tensor] | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compute one-to-many assignment by selecting the top-k lowest-cost queries per ground truth.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape (batch_size, num_queries, 4).
            pred_scores (torch.Tensor): Predicted classification scores with shape (batch_size, num_queries,
                num_classes).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (num_gts, 4).
            gt_cls (torch.Tensor): Ground truth class labels with shape (num_gts,).
            gt_groups (list[int]): Number of ground truth boxes for each image in the batch.
            masks (torch.Tensor, optional): Predicted masks with shape (batch_size, num_queries, height, width).
            gt_mask (list[torch.Tensor], optional): Ground truth masks, each with shape (num_masks, Height, Width).

        Returns:
            (list[tuple[torch.Tensor, torch.Tensor]]): A list of length batch_size. Each element is a tuple
                (query_idx, gt_idx); unlike the Hungarian matcher, a ground truth index may appear up to ``topk`` times.
        """
        bs = pred_scores.shape[0]
        empty = (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
        if sum(gt_groups) == 0:
            return [empty for _ in range(bs)]

        C = self.build_cost_matrix(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks, gt_mask)
        gt_offsets = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)  # global gt index offset per image

        indices = []
        for i, c in enumerate(C.split(gt_groups, -1)):
            cost = c[i]  # (num_queries, num_gt_i)
            n_gt = cost.shape[-1]
            if n_gt == 0:
                indices.append(empty)
                continue
            k = min(self.topk, cost.shape[0])
            # For each ground truth (column) pick the k queries (rows) with the lowest cost.
            top_q = torch.topk(cost, k, dim=0, largest=False).indices  # (k, num_gt_i)
            query_idx = top_q.reshape(-1)
            gt_idx = torch.arange(n_gt).repeat(k) + gt_offsets[i]
            indices.append((query_idx.long(), gt_idx.long()))
        return indices
