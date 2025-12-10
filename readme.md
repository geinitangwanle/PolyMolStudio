调整了生成模型训练/采样脚本的导入为包路径 models.generator.*，并统一在脚本顶部将仓库根目录加入 sys.path；新增 models/generator/__init__.py 以支持包导入。涉及：train/train_generator/train_v1_base.py, train/train_generator/train_v2_base.py, train/train_generator/train_v3_tg.py, train/train_generator/train_v4_pretrain.py, train/train_generator/train_v4_finetune.py, sample/sample_v2_base.py, sample/sample_v3_tg.py, sample/sample_v4_uncond.py, sample/sample_v4_mask.py.
预测模型部分改为包路径 models.predictor.* 和 utils.*，并新增 models/predictor/__init__.py，修正了数据转换脚本导入。涉及：train/train_predictor.py, predict/predict.py, utils/data.py.
可运行命令（仓库根目录执行）：

预训练 v4：python utils/unified_cli.py train --version v4 --mode pretrain -- --csv data/PI1M_v2_psmiles.csv
微调 v4 Tg：python utils/unified_cli.py train --version v4 --mode finetune -- --csv data/PSMILES_Tg_only.csv --pretrained checkpoints/pretrain_modelv4.pt
训练 v3 Tg：python utils/unified_cli.py train --version v3 -- --csv data/PSMILES_Tg_only.csv
训练 v2：python utils/unified_cli.py train --version v2
采样 v4 预训练：python utils/unified_cli.py sample --version v4 --mode uncond -- --checkpoint checkpoints/pretrain_modelv4.pt
采样 v4 Tg：python utils/unified_cli.py sample --version v4 --mode tg -- --checkpoint checkpoints/finetune_tg_modelv4.pt --target-tg 350 450
采样 v3 Tg：python utils/unified_cli.py sample --version v3 -- --checkpoint checkpoints/modelv3_tg.pt --target-tg 350 450
采样 v1/v2：python utils/unified_cli.py sample --version v2 -- --checkpoint checkpoints/modelv2_best.pt
训练预测模型：python train/train_predictor.py --data_path <manifest.csv> --root_dir <npz根目录>
预测推理：python predict/predict.py --ckpt_path <checkpoint.pt> --csv_path <input.csv> --psmiles_col <列名> --save_dir pred_graphs