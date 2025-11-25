from .data import convert_csv_to_graphs
import argparse
'''
使用convert_csv_to_graphs将PSMILES转换为图数据，并保存为.npz文件。
'''
def main():
    parser = argparse.ArgumentParser(description="Convert PSMILES in CSV to graph data and save as .npz files.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to PSMILES CSV.")
    parser.add_argument("--label_col", type=str,  help="Column name for labels in the CSV.")
    parser.add_argument("--PSMILES_col", type=str, required=True, help="Column name for PSMILES in the CSV.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save .npz files.")
    args = parser.parse_args()
    convert_csv_to_graphs(
        csv_path=args.csv_path,  # 输入CSV文件路径
        label_col=args.label_col,
        PSMILES_col=args.PSMILES_col,
        save_dir=args.save_dir       # 保存目录
        )


if __name__ == "__main__":
    main()
