# PolyMolStudio
两个聚合物机器学习工具包的统一仓库：
- `models/polyGeoGAT`：面向 Tg 预测的几何感知 GAT（基于图的性质预测）。
- `models/PolySmith`：带 Tg 条件与语法掩码的 PSMILES 生成器（v1–v4）。

## 仓库结构
- `models/polyGeoGAT/`：图构建、GeoGAT 模型、训练与预测命令行。
- `models/PolySmith/`：VAE/Transformer 生成模型、分版本脚本、统一 CLI 分发器。
- `requirements.txt`：合并后的依赖入口（`pip install -r requirements.txt`）。

## 快速开始
- 安装依赖（建议在虚拟环境内）：
  ```bash
  pip install -r requirements.txt
  ```

- PolyGeoGAT（从仓库根目录运行）：
  ```bash
  python -m models.polyGeoGAT.build_graphs --csv_path models/polyGeoGAT/datasets/Train.csv --label_col label --PSMILES_col PSMILES --save_dir models/polyGeoGAT/graph
  python -m models.polyGeoGAT.train --data_path models/polyGeoGAT/graph/manifest.csv --root_dir models/polyGeoGAT --log_dir models/polyGeoGAT/logs --checkpoint_dir models/polyGeoGAT/checkpoints
  python -m models.polyGeoGAT.predict --ckpt_path models/polyGeoGAT/checkpoints/train_*/best_rmse_*.pt --csv_path models/polyGeoGAT/datasets/Testdataset.csv --psmiles_col PSMILES --out_csv models/polyGeoGAT/preds.csv
  ```

- PolySmith 生成器（从仓库根目录运行）：
  ```bash
  python -m models.PolySmith.unified_cli train --version v4 --mode pretrain -- --csv models/PolySmith/data/PI1M_v2_psmiles.csv
  python -m models.PolySmith.unified_cli train --version v4 --mode finetune -- --csv models/PolySmith/data/PSMILES_Tg_only.csv
  python -m models.PolySmith.unified_cli sample --version v4 --mode tg -- --checkpoint models/PolySmith/checkpoints/finetune_tg_modelv4.pt
  ```

更多细节见各子模块的 README（`models/polyGeoGAT/readme.md`、`models/PolySmith/readme.md`）。

## 单仓多模块骨架
- `models/gnn_predictor/`、`models/psmiles_generator/`：对现有模型的封装入口。
- `libs/`：通用工具（化学/分词/metrics/可视化/dataio）。
- `configs/`：YAML 配置（`gnn/`、`psmiles/`、`common/`、`pipeline.yaml`）。
- `scripts/`：命令行入口：`train_gnn.py`、`eval_gnn.py`、`train_psmiles.py`、`sample_psmiles.py`、`pipeline_generate_and_predict.py`。
- `data/`（raw/processed，小样例即可）、`outputs/`（gnn/psmiles/pipeline）、`tests/`、`docs/`。

示例命令：
```bash
python scripts/train_gnn.py --config configs/gnn/train.yaml
python scripts/eval_gnn.py --config configs/gnn/eval.yaml
python scripts/train_psmiles.py --config configs/psmiles/train.yaml
python scripts/sample_psmiles.py --config configs/psmiles/sample.yaml
python scripts/pipeline_generate_and_predict.py --config configs/pipeline.yaml
```
