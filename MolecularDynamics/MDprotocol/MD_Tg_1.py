#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RadonPy Tg 计算

流程：
1) 从单体 PSMILES 生成 RDKit 分子
2) 单体：构象搜索 + RESP 电荷
3) 终止基：RESP 电荷
4) 用 calc_n_from_num_atoms 估算聚合度 DP，生成一条聚合物链
5) 建立非晶盒 + EQ21step + Additional 平衡
6) 多温度 NpT 冷却，得到 ρ–T 数据
7) 线性分段拟合 ρ–T 曲线，求 Tg
8) 输出：
   - Tg_density_T.csv       （T vs. density）
   - Tg_fit_result.json     （Tg 和拟合参数）
   - Tg_density_T_fit.png.   (Tg曲线拟合图片)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt

from radonpy.core import utils, poly
from radonpy.ff.gaff2_mod import GAFF2_mod
from radonpy.sim import qm
from radonpy.sim.preset import eq   # EQ21step & Additional

# ============================================================
# 0. 全局参数设置（根据自己体系修改）
# ============================================================

# ---- 化学体系 ----
MONOMER_SMILES = "[*]CC([*])OC(=O)C(CCC)(C)C(=O)OC(F)(F)Cl"   # 重复单元的 PSMILES
TER_SMILES     = "*C"               # 终止基 PSMILES，一般要带一个 * 作为连接位点

# ---- MD 条件 ----
T_EQ   = 300.0    # 初始平衡温度 (K)
P_EQ   = 1.0      # 平衡压力 (atm)

# Tg 冷却 protocol
T_HIGH = 700.0    # 冷却起始高温 (K)
T_LOW  = 100.0    # 冷却终止低温 (K)
DT     = -20.0    # 降温步长 (负号表示降温)
TEMP_LIST = np.arange(T_HIGH, T_LOW + DT, DT)   

# ---- 资源配置 ----
MPI_LMP   = 96   # LAMMPS 的 MPI 进程数
OMP_LMP   = 1     # LAMMPS 不用 OpenMP，只走 MPI
GPU_ID    = 0     # 没有 GPU 就设为 0
PSI4_OMP  = 16     # Psi4 线程数（对应 PSI4_NUM_THREADS）
PSI4_MEM  = 20000 # Psi4 内存 (MB)

# ---- 目标体系大小 ----
TARGET_N_ATOMS = 1000  # 需要一条聚合物链大致包含的原子数，用来反算 DP

# ---- 工作目录 ----
WORK_DIR_BASE = "./MD-1"      # 整个项目根目录
QM_DIR        = os.path.join(WORK_DIR_BASE, "qm")#DFT计算结果目录
EQ_DIR        = os.path.join(WORK_DIR_BASE, "eq")#预平衡
TG_DIR        = os.path.join(WORK_DIR_BASE, "tg_cooling")#降温
RESULT_DIR    = os.path.join(WORK_DIR_BASE, "results")   # 存放 Tg 结果的目录

for d in [WORK_DIR_BASE, QM_DIR, EQ_DIR, TG_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

FF = GAFF2_mod()

# ============================================================
# 1. 单体：构象搜索 + RESP 电荷
# ============================================================

def prepare_monomer():
    """对单体做构象搜索 + RESP 电荷拟合"""
    print("=== Step 1: Monomer QM preparation ===")
    # 从 PSMILES 生成 RDKit 分子
    mol = utils.mol_from_smiles(MONOMER_SMILES)

    # 1.1 构象搜索（MM+DFT），得到最低能量构象
    mol, energy = qm.conformation_search(
        mol,
        ff=FF,
        nconf=500,          # ETKDG 生成构象数
        work_dir=QM_DIR,
        psi4_omp=PSI4_OMP,
        mpi=MPI_LMP,
        omp=OMP_LMP,
        memory=PSI4_MEM,
        log_name="monomer_conf"
    )

    # 1.2 在该构象上做 RESP 电荷计算（默认 HF/6-31G*）
    qm.assign_charges(
        mol,
        charge='RESP',
        opt=False,          # 不再几何优化，直接在当前构象上算电荷
        work_dir=QM_DIR,
        omp=PSI4_OMP,
        memory=PSI4_MEM,
        log_name='monomer_charge'
    )
    return mol

# ============================================================
# 2. 终止基团 RESP 电荷
# ============================================================

def prepare_terminator():
    """对终止基做 RESP 电荷"""
    print("=== Step 2: Terminator QM preparation ===")
    ter = utils.mol_from_smiles(TER_SMILES)

    qm.assign_charges(
        ter,
        charge='RESP',
        opt=True,           # 小分子，顺便 DFT 优化一下几何
        work_dir=QM_DIR,
        omp=PSI4_OMP,
        memory=PSI4_MEM,
        log_name='terminator'
    )
    return ter

# ============================================================
# 3. 生成聚合物链
# ============================================================

def build_polymer_chain(monomer, terminator):
    """
    用自避免随机行走生成一条均聚物链：
    1) 用 calc_n_from_num_atoms 反算聚合度 DP
    2) polymerize_rw 生成链
    3) terminate_rw 加上终止基
    """
    print("=== Step 3: Build polymer chain ===")

    #   对于均聚物，只有一个 monomer，终止基通过 terminal1 指定。
    dp = poly.calc_n_from_num_atoms(
        monomer,            # 重复单元
        TARGET_N_ATOMS,     # 目标原子数：希望一条链大致有多少原子
        terminal1=terminator
    )

    print(f"  Estimated DP (degree of polymerization) = {dp}")

    # 生成一条 tacticity='atactic' 的均聚物链
    homopoly = poly.polymerize_rw(
        monomer,
        dp,
        tacticity='atactic'
    )
    # 在链两端加上终止基
    homopoly = poly.terminate_rw(homopoly, terminator)

    return homopoly

# ============================================================
# 4. 力场分配 + 非晶盒 + EQ21step + Additional
# ============================================================

def build_and_equilibrate_cell(homopoly):
    """对聚合物链做 GAFF2 分配 → 建盒 → EQ21step + Additional 平衡"""
    print("=== Step 4: Force field assignment ===")
    result = FF.ff_assign(homopoly)
    if not result:
        raise RuntimeError("[ERROR] Force field assignment failed.")

    print("=== Step 5: Build amorphous cell ===")
    # 用这条链构建非晶盒，这里举例：5 条链、初始低密度 0.05 g/cm3
    ac = poly.amorphous_cell(
        homopoly,
        5,              # 盒子里链的条数，可根据体系大小调整
        density=0.05
    )

    print("=== Step 6: EQ21step equilibration ===")
    # EQ21step 是官方预设的一套 21 步平衡流程（升温、压缩、退火等）
    eqmd = eq.EQ21step(ac, work_dir=EQ_DIR)
    ac = eqmd.exec(
        temp=T_EQ,
        press=P_EQ,
        mpi=MPI_LMP,
        omp=OMP_LMP,
        gpu=GPU_ID,
    )

    analy = eqmd.analyze()
    _ = analy.get_all_prop(temp=T_EQ, press=P_EQ, save=True)
    result_eq = analy.check_eq()
    print(f"  EQ21step check_eq = {result_eq}")

    print("=== Step 7: Additional NpT equilibration (if needed) ===")
    # 如果 EQ21step 还没完全平衡，再追加几轮 Additional NpT
    for i in range(3):#Additional NpT的轮数
        if result_eq:
            print("  System already equilibrated, stop Additional loop.")
            break
        print(f"  Additional round {i + 1}")
        eqmd = eq.Additional(ac, work_dir=EQ_DIR)
        ac = eqmd.exec(
            temp=T_EQ,
            press=P_EQ,
            mpi=MPI_LMP,
            omp=OMP_LMP,
            gpu=GPU_ID
        )
        analy = eqmd.analyze()
        _ = analy.get_all_prop(temp=T_EQ, press=P_EQ, save=True)
        result_eq = analy.check_eq()
        print(f"    check_eq = {result_eq}")

    if not result_eq:
        print("[WARNING] System may not be fully equilibrated at T_EQ.")
    return ac

# ============================================================
# 5. Tg 冷却：在多个温度点做 NpT，收集密度
# ============================================================

def cooling_npt_for_tg(ac_eq):
    """
    在 TEMP_LIST 中的每个温度点做 Additional NpT，
    收集平均密度并返回 T, rho 数组。
    同时把 T-ρ 数据保存到 CSV 文档，便于后续处理。
    """
    print("=== Step 8: NpT cooling for Tg ===")
    T_list = []
    rho_list = []

    current_ac = ac_eq

    for T in TEMP_LIST:
        print(f">>> NpT at {T:.1f} K")

        wd_T = os.path.join(TG_DIR, f"T_{int(T)}K")
        os.makedirs(wd_T, exist_ok=True)

        # 在上一个温度的平衡构型基础上继续降温
        eqmd = eq.Additional(current_ac, work_dir=wd_T)
        current_ac = eqmd.exec(
            temp=float(T),
            press=P_EQ,
            mpi=MPI_LMP,
            omp=OMP_LMP,
            gpu=GPU_ID,
            eq_step=1      # 每个温度点 1 ns NPT!设置NPT时间
        )

        analy = eqmd.analyze()
        ###如果在降温阶段不使用5ns模拟，这里的参数需要调整！！！
        prop = analy.get_all_prop(
            temp=float(T),
            press=P_EQ,
            save=True,   
            init=50,     # 从第 50 帧开始统计
            f_width=50,  # 波动分析窗口
            width=100,   # MSD 和轨迹分析窗口
            )

        # 不同版本的键可能略有不同，这里做几种兼容
        if 'density' in prop:
            rho = prop['density']
        elif 'density_ave' in prop:
            rho = prop['density_ave']
        elif 'spec_volume' in prop:
            rho = 1.0 / prop['spec_volume']
        else:
            print("  [WARNING] Unknown density key, available keys:")
            print(prop.keys())
            raise KeyError("Cannot find density in prop dict.")

        print(f"    density = {rho:.4f} g/cm^3")

        T_list.append(float(T))
        rho_list.append(float(rho))

    # 把 T-ρ 数据写入 CSV，方便后处理/画图
    data = np.column_stack([T_list, rho_list])
    csv_path = os.path.join(RESULT_DIR, "Tg_density_T.csv")
    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header="T(K),density(g/cm^3)",
        comments=""
    )
    print(f"[INFO] Saved T-ρ data to {csv_path}")

    return np.array(T_list), np.array(rho_list), current_ac

# ============================================================
# 6. ρ–T 双线性拟合，计算 Tg，并绘制图片
# ============================================================

def fit_tg_from_rho(T_arr, rho_arr):
    """
    对密度–温度数据做“两段直线拟合”，计算 Tg，
    同时将 Tg 与拟合参数保存为 JSON 文档。
    """
    print("=== Step 9: Fit Tg from rho(T) ===")
    n = len(T_arr)
    if n < 6:
        raise ValueError("Temperature points (<6) are too few for bilinear fit.")

    # 简单划分：前 1/3 为高温段，后 1/3 为低温段
    idx_high = np.arange(0, n // 3)
    idx_low  = np.arange(2 * n // 3, n)

    # 高温段拟合：rho ≈ a1*T + b1
    A_high = np.vstack([T_arr[idx_high], np.ones_like(T_arr[idx_high])]).T
    a1, b1 = np.linalg.lstsq(A_high, rho_arr[idx_high], rcond=None)[0]

    # 低温段拟合：rho ≈ a2*T + b2
    A_low = np.vstack([T_arr[idx_low], np.ones_like(T_arr[idx_low])]).T
    a2, b2 = np.linalg.lstsq(A_low, rho_arr[idx_low], rcond=None)[0]

    # 两条直线交点 = Tg
    Tg = (b2 - b1) / (a1 - a2)

    print("  High-T fit: rho ≈ a1*T + b1, a1 = %.4e, b1 = %.4e" % (a1, b1))
    print("  Low-T  fit: rho ≈ a2*T + b2, a2 = %.4e, b2 = %.4e" % (a2, b2))
    print("  Estimated Tg = %.2f K" % Tg)

    # 把所有拟合信息写入 JSON，方便以后重新分析
    result_dict = {
        "MONOMER_SMILES": MONOMER_SMILES,
        "TER_SMILES": TER_SMILES,
        "T_EQ": T_EQ,
        "P_EQ": P_EQ,
        "T_HIGH": T_HIGH,
        "T_LOW": T_LOW,
        "DT": DT,
        "temperature_list_K": T_arr.tolist(),
        "density_list_g_cm3": rho_arr.tolist(),
        "high_region_indices": idx_high.tolist(),
        "low_region_indices": idx_low.tolist(),
        "fit_params": {
            "a1_high": float(a1),
            "b1_high": float(b1),
            "a2_low": float(a2),
            "b2_low": float(b2)
        },
        "Tg_K": float(Tg)
    }

    json_path = os.path.join(RESULT_DIR, "Tg_fit_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)
    print(f"[INFO] Saved Tg fit parameters to {json_path}")

    return Tg, (a1, b1, a2, b2)
#Tg曲线绘制
def plot_tg_curve(T_arr, rho_arr, Tg, params, out_path):
    """
    根据 T-ρ 数据和双线性拟合结果绘图，并保存为 out_path（PNG 等）。
    params = (a1, b1, a2, b2)
    """
    a1, b1, a2, b2 = params

    # 交点处密度
    rho_Tg = a1 * Tg + b1

    # 为了画出两条完整直线，构造一个平滑的温度范围
    T_fit = np.linspace(T_arr.min() - 5, T_arr.max() + 5, 200)
    rho_high_fit = a1 * T_fit + b1
    rho_low_fit  = a2 * T_fit + b2

    plt.figure(figsize=(6, 4.5), dpi=150)

    # 所有 MD 数据点
    plt.scatter(T_arr, rho_arr, color="black", s=35, zorder=3, label="MD data")

    # 高温段拟合直线
    plt.plot(T_fit, rho_high_fit, linestyle="--", linewidth=2.0,
             label="High-T fit")

    # 低温段拟合直线
    plt.plot(T_fit, rho_low_fit, linestyle="--", linewidth=2.0,
             label="Low-T fit")

    # Tg 竖线 + 交点
    plt.axvline(Tg, color="red", linestyle=":", linewidth=2.0)
    plt.scatter([Tg], [rho_Tg], color="red", s=50, zorder=4)
    plt.text(Tg + 3, rho_Tg + 0.003,
             f"Tg ≈ {Tg:.1f} K",
             color="red", fontsize=11)

    # 轴标签和标题（全英文、期刊风）
    plt.xlabel("Temperature / K", fontsize=12)
    plt.ylabel("Density / g·cm$^{-3}$", fontsize=12)
    plt.title("Density–temperature curve and Tg", fontsize=14)

    plt.xlim(T_arr.min() - 10, T_arr.max() + 10)
    plt.ylim(rho_arr.min() - 0.02, rho_arr.max() + 0.02)

    plt.legend(frameon=True, fontsize=10)
    plt.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()

    # 只保存，不 show（超算上也不会弹窗）
    plt.savefig(out_path)
    plt.close()


# ============================================================
# 7. 主程序
# ============================================================

def main():
    monomer    = prepare_monomer()
    terminator = prepare_terminator()
    homopoly   = build_polymer_chain(monomer, terminator)
    ac_eq      = build_and_equilibrate_cell(homopoly)

    T_arr, rho_arr, ac_final = cooling_npt_for_tg(ac_eq)
    Tg, params = fit_tg_from_rho(T_arr, rho_arr)

    # === 画图并保存到 results 目录 ===
    fig_path = os.path.join(RESULT_DIR, "Tg_density_T_fit.png")
    plot_tg_curve(T_arr, rho_arr, Tg, params, fig_path)
    print("  - Tg_density_T_fit.png")

    print("====================================")
    print("Final estimated Tg (MD) = %.2f K" % Tg)
    print("Results saved in directory:", RESULT_DIR)
    print("  - Tg_density_T.csv")
    print("  - Tg_fit_result.json")
    print("  - Tg_density_T_fit.png")
    print("====================================")



if __name__ == "__main__":
    main()
