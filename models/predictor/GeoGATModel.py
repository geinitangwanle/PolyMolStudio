# ConvModel.py 2025/09/23
import math
import warnings
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import global_mean_pool, GATv2Conv
from torch_geometric.utils import to_dense_batch
from typing import Optional

from .GatedConv import GatedGraphConv
from .GeomFeat import GeometryFeaturizer, GemNetEdgeUpdate

class GeoGATModel(torch.nn.Module):
    def __init__(
        self,
        layers_in_conv=3, # 每个卷积块中图卷积层的个数为3，代表消息传递的次数
        channels=64, # 图卷积层中的隐藏维度
        use_nodetype_coeffs=True, 
        num_node_types=9, # 节点类型个数
        num_edge_types=4, # 边类型个数
        use_jumping_knowledge=False,
        embedding_size=64,
        use_bias_for_update=True,
        use_dropout=True,
        num_convs=3, # 模型中的卷积块个数为3
        num_fc_layers=3,
        neighbors_aggr='add', # 节点特征聚合方式
        dropout_p=0.1,
        num_targets=1,
        # =========================== 新增接口 =====================
        geom_K=16, # RBF探针数量
        geom_rmax=4.0, # 设置化学键的最大键长
        concat_original_edge=True, # 得到新的边特征后，是否与原始边特征拼接（是否使用pos信息）
        gem_out=32,
        heads=4, # GAT多头注意力机制的头数
        # ============== 融合 pSMILES (PolyBERT) 的 cross-attention ==============
        use_polybert: bool = False,
        polybert_name: str = "kuelumbus/polyBERT",
        polybert: Optional[nn.Module] = None,
        freeze_polybert: bool = True,
        seq_max_length: int = 256,
        cross_attn_heads: int = 4,
        cross_attn_dim: Optional[int] = None,  # 若为 None 则默认等于 channels
    ):
        super(GeoGATModel, self).__init__()

        self.num_convs = num_convs
        self.layers_in_conv = layers_in_conv
        self.channels = channels
        self.use_jumping_knowledge = use_jumping_knowledge
        self.use_dropout = use_dropout
        self.pool_to_channels = nn.Linear(64, 128) # 线性层，用于将池化后的特征映射到128维
        self.geom_K = geom_K
        self.geom_rmax = geom_rmax
        self.concat_original_edge = concat_original_edge
        self.use_polybert = use_polybert
        self.polybert_name = polybert_name
        self.freeze_polybert = freeze_polybert
        self.seq_max_length = seq_max_length
        self.cross_attn_heads = cross_attn_heads
        self.cross_attn_dim = cross_attn_dim or channels
        # 兜底：保证 embed_dim 可整除 heads；若不能整除，优先调小 heads，否则调大 dim
        if self.cross_attn_dim % self.cross_attn_heads != 0:
            # 找到 cross_attn_dim 的最大可用因子，确保 heads <= 原 heads
            factors = [h for h in range(self.cross_attn_heads, 0, -1) if self.cross_attn_dim % h == 0]
            if factors:
                new_heads = factors[0]
                warnings.warn(
                    f"cross_attn_dim={self.cross_attn_dim} not divisible by heads={self.cross_attn_heads}; "
                    f"using heads={new_heads} instead."
                )
                self.cross_attn_heads = new_heads
            else:
                rounded = self.cross_attn_heads * math.ceil(self.cross_attn_dim / self.cross_attn_heads)
                warnings.warn(
                    f"cross_attn_dim={self.cross_attn_dim} not divisible by heads={self.cross_attn_heads}; "
                    f"rounding embed_dim up to {rounded}."
                )
                self.cross_attn_dim = rounded

        # === 几何特征编码器 ===
        self.geom = GeometryFeaturizer(K=geom_K, r_min=0.0, r_max=geom_rmax,concat_original=concat_original_edge) # 边RBF

        self.gem_edge = GemNetEdgeUpdate(K_r=geom_K, r_min=0.0, r_max=geom_rmax, # 角度
                                         K_a=8, mlp_hidden=64, out_dim=gem_out, aggr='add')

        # 原始4 + 距离K + 角增强 gem_out
        edge_in_dim = (4 + geom_K + gem_out) if concat_original_edge else 4


        self.mggc1 = GatedGraphConv(
            out_channels=channels,
            num_layers=layers_in_conv,
            num_edge_types=num_edge_types,
            num_node_types=num_node_types,
            aggr=neighbors_aggr,
            edge_in_size=edge_in_dim,
            use_nodetype_coeffs=False,
            use_jumping_knowledge=False,
            use_bias_for_update=True,
            node_in_dim=4,
        )

        self.gat2 = GATv2Conv(
            in_channels=channels,
            out_channels=channels // heads,  # concat=True 时 head*out = channels
            heads=heads,
            concat=True,
            edge_dim=edge_in_dim,            # 关键：把 (距离/角度/原始边) 送进注意力
            dropout=0.1,                     # 注意力 dropout
            add_self_loops=True
        )

        self.bn_gat2 = nn.BatchNorm1d(channels)
        self.res_gat2 = nn.Linear(channels, channels, bias=False)  # 可选残差对齐（若维度一致也可直接恒等）

        self.mggc3 = GatedGraphConv(
            out_channels=channels,
            num_layers=layers_in_conv,
            num_edge_types=num_edge_types,
            num_node_types=num_node_types,
            aggr=neighbors_aggr,
            edge_in_size=edge_in_dim,
            use_nodetype_coeffs=False,
            use_jumping_knowledge=False,
            use_bias_for_update=True,
            node_in_dim=channels,
        )

        #set2set全局池化层，使用了LSTM
        self.set2set = pyg.nn.Set2Set(channels, processing_steps=5, num_layers=2)

        # 标准化，用于每个卷积块前
        self.batch_norms = nn.ModuleList(
            [torch.nn.BatchNorm1d(channels) for _ in range(num_convs)]
        )

        # 暂退
        self.dropout = nn.Dropout(p=dropout_p)

        # ============== PolyBERT & Cross-Attention ==============
        self.polybert_hidden_dim = None
        if self.use_polybert:
            if polybert is not None:
                self.polybert = polybert
            else:
                try:
                    from transformers import AutoModel
                except ImportError as e:  # pragma: no cover
                    raise ImportError("transformers is required for use_polybert=True") from e
                self.polybert = AutoModel.from_pretrained(polybert_name)
            if self.freeze_polybert:
                for p in self.polybert.parameters():
                    p.requires_grad = False
                self.polybert.eval()
            self.polybert_hidden_dim = getattr(self.polybert.config, "hidden_size", None)
            if self.polybert_hidden_dim is None:
                raise ValueError("polyBERT model must expose hidden_size in config.")

            self.seq_proj = nn.Linear(self.polybert_hidden_dim, self.cross_attn_dim)
            self.graph_proj = nn.Linear(channels, self.cross_attn_dim)
            self.cross_attn_seq_to_graph = nn.MultiheadAttention(
                embed_dim=self.cross_attn_dim, num_heads=cross_attn_heads, batch_first=True
            )
            self.cross_attn_graph_to_seq = nn.MultiheadAttention(
                embed_dim=self.cross_attn_dim, num_heads=cross_attn_heads, batch_first=True
            )

        # FC 输入维度：原始 3*channels + （可选 cross-attn 拼接）
        base_fc_in = 3 * self.channels
        cross_in = 2 * self.cross_attn_dim if self.use_polybert else 0
        self.fc_in_dim = base_fc_in + cross_in

        # 构建多层线性层
        self.fc_layers = nn.ModuleList(
            self.make_fc_layers(num_fc_layers, num_targets=num_targets)
        )

        # 在线性层之前加一个标准化层
        self.pre_fc_batchnorm = torch.nn.BatchNorm1d(self.fc_layers[0].in_features)

        # 为每一个线性层之后加一个标准化层
        self.batch_norms_for_fc = nn.ModuleList(
            [
                torch.nn.BatchNorm1d(self.fc_layers[i + 1].in_features)
                for i in range(num_fc_layers - 1)
            ]
        )

    # 用于构建线性层的功能函数
    def make_fc_layers(self, num_fc_layers, num_targets):
        fc_layers = []
        in_channels = self.fc_in_dim
        for i in range(num_fc_layers):
            out_channels = num_targets if i == num_fc_layers - 1 else max(in_channels // 2, 8)
            fc_layers.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels
        return fc_layers

    @staticmethod
    def _masked_mean(tensor, mask):
        """
        tensor: [B, L, D]
        mask:   [B, L] (bool or 0/1)，True 表示有效位
        """
        mask_f = mask.unsqueeze(-1).to(tensor.dtype)
        denom = mask_f.sum(dim=1).clamp(min=1e-6)
        return (tensor * mask_f).sum(dim=1) / denom

    def export_config(self):
        return {
            "layers_in_conv": self.layers_in_conv,
            "channels": self.channels,
            "use_nodetype_coeffs": False,
            "num_node_types": 0,
            "num_edge_types": 4,
            "use_jumping_knowledge": False,
            "use_bias_for_update": True,
            "use_dropout": self.use_dropout,
            "num_convs": self.num_convs,
            "num_fc_layers": len(self.fc_layers),
            "neighbors_aggr": "add",
            "dropout_p": self.dropout.p if hasattr(self, "dropout") else 0.0,
            "num_targets": self.fc_layers[-1].out_features if self.fc_layers else 1,
            "geom_K": self.geom_K,
            "geom_rmax": self.geom_rmax,
            "concat_original_edge": self.concat_original_edge,
            "use_polybert": self.use_polybert,
            "polybert_name": self.polybert_name,
            "freeze_polybert": self.freeze_polybert,
            "seq_max_length": self.seq_max_length,
            "cross_attn_heads": self.cross_attn_heads,
            "cross_attn_dim": self.cross_attn_dim,
        }

    # 前向传播部分
    def forward(self, data):
        x, edge_index, edge_attr, pos, batch = data.x, data.edge_index, data.edge_attr, data.pos, data.batch
        batch_size = int(batch.max().item()) + 1

        # (1) 距离 RBF（+原始边）
        edge_rbf = self.geom(pos, edge_index, edge_attr)  # [E, 4+K] 或 [E, K]

        # (2) 角三元组 → 边增强
        edge_trip = self.gem_edge(pos, edge_index)        # [E, gem_out]

        # (3) 拼接成最终边特征
        edge_attr_all = torch.cat([edge_rbf, edge_trip], dim=-1)  # [E, base_dim + gem_out]

        # block-1: 仍用门控卷积
        x = self.mggc1(x, edge_index, edge_attr_all); x = self.batch_norms[0](x); x = F.relu(x)

        # block-2: GATv2（带边特征）
        x_res = x
        x = self.gat2(x, edge_index, edge_attr_all)         # [N, channels]
        x = self.bn_gat2(x)
        x = F.relu(x)
        x = x + self.res_gat2(x_res) 

        # block-3: 仍用门控卷积
        x = self.mggc3(x, edge_index, edge_attr_all); x = self.batch_norms[2](x); x = F.relu(x)

        seq_pooled = graph_pooled = None
        if self.use_polybert and hasattr(data, "seq_input_ids") and hasattr(data, "seq_attention_mask"):
            # 将 batch 中的节点特征补齐成 [B, N_max, C]
            graph_dense, graph_mask = to_dense_batch(x, batch)  # mask: True 表示有效节点

            seq_ids = data.seq_input_ids.view(batch_size, self.seq_max_length)
            seq_mask = data.seq_attention_mask.view(batch_size, self.seq_max_length)
            seq_out = self.polybert(input_ids=seq_ids, attention_mask=seq_mask)
            seq_hidden = seq_out.last_hidden_state  # [B, L, H_poly]

            seq_repr = self.seq_proj(seq_hidden)          # [B, L, D]
            graph_repr = self.graph_proj(graph_dense)     # [B, N, D]

            graph_pad = ~graph_mask  # MultiheadAttention 需要 padding 为 True
            seq_pad = (seq_mask == 0)

            # seq → graph: 查询序列，键值图节点
            seq_ctx, _ = self.cross_attn_seq_to_graph(
                query=seq_repr,
                key=graph_repr,
                value=graph_repr,
                key_padding_mask=graph_pad,
            )
            # graph → seq: 查询节点，键值序列
            graph_ctx, _ = self.cross_attn_graph_to_seq(
                query=graph_repr,
                key=seq_repr,
                value=seq_repr,
                key_padding_mask=seq_pad,
            )

            seq_pooled = self._masked_mean(seq_ctx, seq_mask.bool())     # [B, D]
            graph_pooled = self._masked_mean(graph_ctx, graph_mask)      # [B, D]

        x_1 = self.set2set(x, batch)

        x_2 = global_mean_pool(x, batch) # 使用全局池化

        features = [x_1, x_2]
        if self.use_polybert:
            if seq_pooled is None or graph_pooled is None:
                zeros = x.new_zeros((batch_size, self.cross_attn_dim))
                seq_pooled = zeros if seq_pooled is None else seq_pooled
                graph_pooled = zeros if graph_pooled is None else graph_pooled
            features.extend([seq_pooled, graph_pooled])

        x = torch.cat(features, dim=1) # 拼接融合后的特征

        x = self.pre_fc_batchnorm(x)

        for i, fc in enumerate(self.fc_layers):
            if self.use_dropout and i == 1: # 只在第一个线性层进行dropout
                x = self.dropout(x)
            x = fc(x)
            if i != len(self.fc_layers) - 1:
                x = self.batch_norms_for_fc[i](x)
                x = F.relu(x)

        return x
