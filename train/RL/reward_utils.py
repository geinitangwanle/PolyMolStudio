"""通用奖励计算：validity/novelty/SA + 目标 Tg 条件值。

说明
- 默认将条件 Tg（已标准化）作为正向奖励；并乘上合法性、SA、独特性系数。
- 依赖 RDKit 以获得有效性和 SA；若未安装 RDKit，则退回默认值（valid=1, sa=0.5）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set

import torch


@dataclass
class RewardConfig:
    w_tg: float = 1.0         # 条件 Tg（或预测 Tg）权重
    w_valid: float = 0.5      # 合法性权重
    w_sa: float = 0.25        # 合成可行性权重
    w_novelty: float = 0.25   # 新颖性权重
    sa_floor: float = 0.0     # SA 评分下界
    sa_ceil: float = 10.0     # SA 评分上界
    novelty_set: Set[str] = field(default_factory=set)  # 已见集合，用于新颖性


def _try_import_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import QED
        from rdkit.Chem import rdMolDescriptors
        return Chem, QED, rdMolDescriptors
    except Exception:
        return None, None, None


def decode_tokens(tokenizer, tokens: torch.Tensor) -> List[str]:
    # tokens: [B, T]
    decoded = []
    for seq in tokens.tolist():
        try:
            decoded.append(tokenizer.decode(seq, skip_special_tokens=True))
        except Exception:
            decoded.append("")
    return decoded


def _validity_sa(smiles_list: List[str], cfg: RewardConfig):
    Chem, QED, rdMolDescriptors = _try_import_rdkit()
    valid_mask = []
    sa_scores = []
    if Chem is None:
        # RDKit 不可用时，默认有效，SA=0.5
        for s in smiles_list:
            valid_mask.append(1.0 if s else 0.0)
            sa_scores.append(0.5)
        return torch.tensor(valid_mask), torch.tensor(sa_scores)

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            valid_mask.append(0.0)
            sa_scores.append(0.0)
            continue
        valid_mask.append(1.0)
        try:
            sa_raw = rdMolDescriptors.CalcNumRings(mol) + rdMolDescriptors.CalcExactMolWt(mol) * 0.0  # lightweight
            # 若有 SA_Score 可用，可替换此逻辑；这里简单用 QED 的反向作为可合成度近似
            sa_raw = 1.0 - QED.qed(mol)
        except Exception:
            sa_raw = 0.5
        sa_scores.append(sa_raw)

    return torch.tensor(valid_mask), torch.tensor(sa_scores)


def _normalize_sa(sa: torch.Tensor, cfg: RewardConfig):
    # 将 SA 评分映射到 [0,1]，分值越小越好（更易合成）
    sa_clipped = sa.clamp(cfg.sa_floor, cfg.sa_ceil)
    return 1.0 - (sa_clipped - cfg.sa_floor) / max(cfg.sa_ceil - cfg.sa_floor, 1e-6)


def _novelty(smiles_list: List[str], cfg: RewardConfig):
    novelty = []
    for s in smiles_list:
        if not s:
            novelty.append(0.0)
            continue
        is_new = 1.0 if s not in cfg.novelty_set else 0.0
        novelty.append(is_new)
        cfg.novelty_set.add(s)
    return torch.tensor(novelty)


def compute_reward(
    tokens: torch.Tensor,
    conditions: torch.Tensor,
    tokenizer,
    cfg: RewardConfig,
    *,
    tg_predict_fn=None,
    match_target: bool = False,
) -> torch.Tensor:
    """综合奖励：
    reward = w_tg * tg_term + w_valid * valid + w_sa * sa_norm + w_novelty * novelty

    - 默认 tg_term=conditions（数据中提供的 Tg，已标准化）。
    - 若传入 tg_predict_fn(smiles_list)->Tensor，则使用预测 Tg：
        * match_target=True: tg_term = -|pred - target|（越接近条件 Tg 越好）
        * match_target=False: tg_term = pred（直接鼓励高预测 Tg）
    - valid/sa/novelty 需 RDKit 才能得到更可靠的值；否则退回默认。
    返回形状 [batch]。
    """

    device = tokens.device
    smiles_list = decode_tokens(tokenizer, tokens)

    valid_mask, sa_raw = _validity_sa(smiles_list, cfg)
    novelty = _novelty(smiles_list, cfg)
    sa_norm = _normalize_sa(sa_raw, cfg)

    if tg_predict_fn is not None:
        tg_pred = torch.as_tensor(tg_predict_fn(smiles_list), device=device, dtype=torch.float32)
        tg_target = conditions.view(-1).to(device)
        if match_target:
            tg_term = -torch.abs(tg_pred - tg_target)
        else:
            tg_term = tg_pred
    else:
        tg_term = conditions.view(-1).to(device)

    reward = (
        cfg.w_tg * tg_term
        + cfg.w_valid * valid_mask.to(device)
        + cfg.w_sa * sa_norm.to(device)
        + cfg.w_novelty * novelty.to(device)
    )
    return reward
