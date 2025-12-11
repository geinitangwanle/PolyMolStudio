"""REINVENT / DPO losses and helper utilities.

核心：给定 logp_agent、logp_prior 与 reward，返回标量 loss，prior 永不训练。
"""

from __future__ import annotations

from typing import Literal

import torch


def reinvent_loss(
    logp_agent: torch.Tensor,
    logp_prior: torch.Tensor,
    reward: torch.Tensor,
    *,
    reduction: Literal["mean", "sum", "none"] = "mean",
):
    """REINVENT: L = -(logp_agent - logp_prior) * R."""

    loss = -((logp_agent - logp_prior) * reward)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def dpo_loss(
    logp_agent: torch.Tensor,
    logp_prior: torch.Tensor,
    reward: torch.Tensor,
    *,
    sigma: float = 1.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
):
    """DPO (OPV 形式): L = (logp_prior - logp_agent + sigma * R)^2."""

    margin = logp_prior - logp_agent + sigma * reward
    loss = margin.pow(2)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


@torch.no_grad()
def sequence_logp(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
    """调用模型 log_prob，返回 (seq_logp, token_logp)。"""

    return model.log_prob(input_ids, attention_mask)


def rl_step(
    *,
    agent,
    prior,
    optimizer,
    reward: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    method: Literal["reinvent", "dpo"] = "dpo",
    sigma: float = 1.0,
):
    """单步 RL 更新：计算 loss 反向并优化 agent，prior 冻结。"""

    logp_agent, _ = agent.log_prob(input_ids, attention_mask)
    with torch.no_grad():
        logp_prior, _ = prior.log_prob(input_ids, attention_mask)

    if method == "reinvent":
        loss = reinvent_loss(logp_agent, logp_prior, reward, reduction="mean")
    else:
        loss = dpo_loss(logp_agent, logp_prior, reward, sigma=sigma, reduction="mean")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {
        "loss": loss.detach(),
        "logp_agent": logp_agent.detach(),
        "logp_prior": logp_prior.detach(),
    }
