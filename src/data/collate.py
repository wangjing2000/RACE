from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

def race_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_text = max(item["text_feats"].shape[0] for item in batch)
    max_region = max(item["region_feats"].shape[0] for item in batch)
    max_ocr = max(item["ocr_feats"].shape[0] for item in batch)

    d_c = batch[0]["h_t_global"].shape[0]

    h_t_global_list, h_v_global_list = [], []
    text_feats_list, text_mask_list = [], []
    region_feats_list, region_mask_list = [], []
    ocr_feats_list, ocr_mask_list = [], []
    labels = []
    ids = []

    for item in batch:
        h_t_global_list.append(item["h_t_global"])
        h_v_global_list.append(item["h_v_global"])
        labels.append(item["label"])
        ids.append(item["id"])

        t_feat = item["text_feats"]
        t_mask = item["text_mask"]
        n_t = t_feat.shape[0]
        if n_t < max_text:
            pad_t = torch.zeros(max_text - n_t, d_c, dtype=t_feat.dtype)
            text_feats_list.append(torch.cat([t_feat, pad_t], dim=0))
            text_mask_list.append(torch.cat([t_mask, torch.zeros(max_text - n_t, dtype=torch.bool)]))
        else:
            text_feats_list.append(t_feat)
            text_mask_list.append(t_mask)

        r_feat = item["region_feats"]
        r_mask = item["region_mask"]
        n_r = r_feat.shape[0]
        if n_r < max_region:
            pad_r = torch.zeros(max_region - n_r, d_c, dtype=r_feat.dtype)
            region_feats_list.append(torch.cat([r_feat, pad_r], dim=0))
            region_mask_list.append(torch.cat([r_mask, torch.zeros(max_region - n_r, dtype=torch.bool)]))
        else:
            region_feats_list.append(r_feat)
            region_mask_list.append(r_mask)

        o_feat = item["ocr_feats"]
        o_mask = item["ocr_mask"]
        n_o = o_feat.shape[0]
        if n_o < max_ocr:
            pad_o = torch.zeros(max_ocr - n_o, d_c, dtype=o_feat.dtype)
            ocr_feats_list.append(torch.cat([o_feat, pad_o], dim=0))
            ocr_mask_list.append(torch.cat([o_mask, torch.zeros(max_ocr - n_o, dtype=torch.bool)]))
        else:
            ocr_feats_list.append(o_feat)
            ocr_mask_list.append(o_mask)

    return {
        "h_t_global": torch.stack(h_t_global_list),
        "h_v_global": torch.stack(h_v_global_list),
        "text_feats": torch.stack(text_feats_list),
        "text_mask": torch.stack(text_mask_list),
        "region_feats": torch.stack(region_feats_list),
        "region_mask": torch.stack(region_mask_list),
        "ocr_feats": torch.stack(ocr_feats_list),
        "ocr_mask": torch.stack(ocr_mask_list),
        "label": torch.tensor(labels, dtype=torch.long),
        "id": ids,
    }

def build_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=race_collate_fn,
        drop_last=False,
    )