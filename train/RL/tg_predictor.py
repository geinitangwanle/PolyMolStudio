"""轻量级 Tg 图预测器封装（GeoGATModel），可直接在 RL 中调用。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from transformers import AutoModel

from models.predictor.GeoGATModel import GeoGATModel
from models.generator.tokenizer import PolyBertTokenizer
from utils.PSMILES_to_graph import graph_from_psmiles


def _denorm(t: torch.Tensor, y_mean: Optional[torch.Tensor], y_std: Optional[torch.Tensor]) -> torch.Tensor:
    if y_mean is None or y_std is None:
        return t
    return t * y_std + y_mean


class TgGraphPredictor:
    """从 GeoGAT checkpoint 加载模型，对一批 SMILES 返回 Tg 预测。"""

    def __init__(self, ckpt_path: str | Path, device: torch.device, batch_size: int = 64):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Tg checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        use_polybert = cfg.get("use_polybert", False)

        self.tokenizer = PolyBertTokenizer(cfg.get("polybert_name", "kuelumbus/polyBERT")) if use_polybert else None
        polybert_model = AutoModel.from_pretrained(cfg["polybert_name"]) if use_polybert else None
        if use_polybert and cfg.get("freeze_polybert", True) and polybert_model is not None:
            for p in polybert_model.parameters():
                p.requires_grad = False
            polybert_model.eval()

        self.model = GeoGATModel(
            layers_in_conv=cfg.get("layers_in_conv", 3),
            channels=cfg.get("channels", 64),
            use_nodetype_coeffs=cfg.get("use_nodetype_coeffs", False),
            num_node_types=cfg.get("num_node_types", 0),
            num_edge_types=cfg.get("num_edge_types", 4),
            use_jumping_knowledge=cfg.get("use_jumping_knowledge", False),
            use_bias_for_update=cfg.get("use_bias_for_update", True),
            use_dropout=cfg.get("use_dropout", True),
            num_convs=cfg.get("num_convs", 3),
            num_fc_layers=cfg.get("num_fc_layers", 3),
            neighbors_aggr=cfg.get("neighbors_aggr", "add"),
            dropout_p=cfg.get("dropout_p", 0.1),
            num_targets=cfg.get("num_targets", 1),
            geom_K=cfg.get("geom_K", 16),
            geom_rmax=cfg.get("geom_rmax", 4.0),
            concat_original_edge=cfg.get("concat_original_edge", True),
            use_polybert=use_polybert,
            polybert=polybert_model,
            freeze_polybert=cfg.get("freeze_polybert", True),
            seq_max_length=cfg.get("seq_max_length", 256),
            cross_attn_heads=cfg.get("cross_attn_heads", 4),
            cross_attn_dim=cfg.get("cross_attn_dim", None),
        ).to(device)

        state_dict = None
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state") or checkpoint.get("state_dict")
        if state_dict is None:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.y_mean = None
        self.y_std = None
        if isinstance(checkpoint, dict):
            if "y_mean" in checkpoint:
                self.y_mean = torch.tensor(checkpoint["y_mean"], dtype=torch.float32, device=device)
            if "y_std" in checkpoint:
                self.y_std = torch.tensor(checkpoint["y_std"], dtype=torch.float32, device=device)

        self.device = device
        self.batch_size = batch_size

    def _smiles_to_data(self, smiles_list: List[str]) -> List[Data]:
        data_list = []
        for idx, smi in enumerate(smiles_list):
            try:
                g = graph_from_psmiles(smi)
                node_feats = g["node_feats"].to_numpy()
                edge_index = torch.as_tensor(g["edge_index"].to_numpy(), dtype=torch.long)
                edge_attr = torch.as_tensor(g["edge_attr"].to_numpy(), dtype=torch.float32)
                x = torch.as_tensor(node_feats[:, 0:4], dtype=torch.float32)
                pos = torch.as_tensor(node_feats[:, 4:7], dtype=torch.float32)
                data = Data(
                    x=x,
                    pos=pos,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    mol_id=idx,
                )
                if self.tokenizer is not None:
                    tokens = self.tokenizer.tokenizer(
                        smi,
                        padding="max_length",
                        truncation=True,
                        max_length=self.model.seq_max_length if hasattr(self.model, "seq_max_length") else 256,
                        return_tensors="pt",
                    )
                    data.seq_input_ids = tokens["input_ids"].squeeze(0)
                    data.seq_attention_mask = tokens["attention_mask"].squeeze(0)
                data_list.append(data)
            except Exception:
                continue
        return data_list

    @torch.no_grad()
    def __call__(self, smiles_list: List[str]) -> torch.Tensor:
        if len(smiles_list) == 0:
            return torch.empty(0, device=self.device)

        data_list = self._smiles_to_data(smiles_list)
        if len(data_list) == 0:
            return torch.zeros(len(smiles_list), device=self.device)

        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)
        preds = torch.zeros(len(smiles_list), device=self.device)

        self.model.eval()
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch).view(-1)
            out = _denorm(out, self.y_mean, self.y_std)
            mol_ids = batch.mol_id
            for i, mid in enumerate(mol_ids):
                idx = int(mid.item())
                preds[idx] = out[i]
        return preds
