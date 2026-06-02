from MDAnalysis import *
import MDAnalysis.analysis.distances as distances
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# ============================================================
# --- 引入分析資料 ---
# ============================================================
topology = "start_drudes.pdb"
trajectory = "FV_NVT.dcd"
electrode_pdb = "start_nodrudes.pdb"  # 用於偵測電極邊界位置
u = Universe(topology, trajectory)

# ============================================================
# --- 參數設定區 ---
# ============================================================
n_bins    = 1000    # Z 軸 histogram bin 數（越多越細緻）
framestart = len(u.trajectory)//2  # 從中間開始分析，避免初始幀的非平衡狀態
frameend   = len(u.trajectory)  # slice exclusive

# Z 軸範圍：使用整個盒子（完整分佈）
# 注意：盒子尺寸可能隨幀微變（NPT），這裡用第一幀的值初始化
u.trajectory[0]
z_box = u.dimensions[2]  # 盒子 Z 方向長度

print(f"📦 盒子 Z 方向長度: {z_box:.3f} Å")
print(f"   bin 寬度: {z_box / n_bins:.3f} Å\n")

# ============================================================
# --- 定義 5 個官能團 ---
# ============================================================
print("🎯 正在定義 5 個官能團...")

functional_groups = {
    # 陰離子 TFSI
    'SO2_polar'     : {
        'sel'  : "resname Tf2 and name Otf Otf1 Otf2 Otf3",
        'label': 'TFSI $-SO_2$ (Polar)',
        'color': '#d62728',   # 紅
        'ls'   : '-'
    },
    'CF3_nonpolar'  : {
        'sel'  : "resname Tf2 and name Ctf Ctf1",
        'label': 'TFSI $-CF_3$ (Non-polar)',
        'color': '#1f77b4',   # 藍
        'ls'   : '-'
    },
    # 陽離子 BMIM
    'Im_polar'      : {
        'sel'  : "resname BMI and name C1 C2 C21",
        'label': 'BMIM Imidazolium Ring (Polar)',
        'color': '#2ca02c',   # 綠
        'ls'   : '-'
    },
    'CH3_nonpolar'  : {
        'sel'  : "resname BMI and name C3",
        'label': 'BMIM $-CH_3$ (Non-polar)',
        'color': '#ff7f0e',   # 橘
        'ls'   : '-'
    },
    'Butyl_nonpolar': {
        'sel'  : "resname BMI and name C4 C5 C51 C6",
        'label': 'BMIM $-C_4H_9$ (Non-polar)',
        'color': '#9467bd',   # 紫
        'ls'   : '-'
    }
}

# ============================================================
# --- 【加速改動一】：select_atoms 移到迴圈外，只做一次 ---
#
# 原子數固定（NVT），所以每幀的原子群組完全相同。
# MDAnalysis 的 AtomGroup.positions 會在每次存取時
# 自動讀取當前幀的座標，不需要重新 select_atoms。
# ============================================================
print("⚡ 預先建立原子群組（只做一次）...")
atom_groups = {
    name: u.select_atoms(info['sel'])
    for name, info in functional_groups.items()
}
for name, grp in atom_groups.items():
    print(f"   {name}: {len(grp)} 個原子")
print()

# ============================================================
# --- 資料結構初始化 ---
# ============================================================
# 每個官能團一個 histogram，統計 z 座標分佈
histograms = {
    name: np.zeros(n_bins, dtype=np.float64)
    for name in functional_groups
}

n_frames = 0

# ============================================================
# --- 軌跡迴圈 ---
# ============================================================
print(f"📊 開始分析軌跡: frame {framestart} → {frameend - 1}")
print(f"   總共約 {frameend - framestart} 幀\n")

for ts in tqdm(u.trajectory[framestart:frameend], desc="Processing frames", unit="frame", ncols=80):
    n_frames += 1

    # 每幀的盒子 Z 長度（應對 NPT 系綜的盒子微變）
    z_box_frame = u.dimensions[2]

    for name, info in functional_groups.items():
        grp = u.select_atoms(info['sel'])

        if len(grp) == 0:
            continue

        # 取得所有原子的 z 座標
        z_coords = grp.positions[:, 2]

        # 統計 histogram（範圍為整個盒子 Z 方向）
        counts, _ = np.histogram(z_coords, bins=n_bins, range=(0.0, z_box_frame))
        histograms[name] += counts

print(f"\n✅ 軌跡運算完成！實際處理幀數: {n_frames}")

# ============================================================
# --- 正規化：計算數密度 ρ(z) ---
#
# ρ(z) = <N(z)> / (A × Δz)
#
# 其中：
#   <N(z)>  = 平均每幀在 z 處的 bin 內的原子數 = histogram / n_frames
#   A       = 盒子的 XY 截面積
#   Δz      = bin 寬度
#
# 單位：原子數 / Å³
# ============================================================
print("\n⏳ 開始進行正規化與寫入 .dat 檔案...")

# 使用第一幀的盒子尺寸計算截面積與 bin 寬
u.trajectory[0]
A_xy  = u.dimensions[0] * u.dimensions[1]  # XY 截面積 (Å²)
dz    = z_box / n_bins                       # bin 寬度 (Å)
z_centers = np.linspace(dz / 2, z_box - dz / 2, n_bins)  # 各 bin 中心的 z 值

print(f"   XY 截面積: {A_xy:.3f} Å²")
print(f"   bin 寬度 Δz: {dz:.4f} Å\n")

density_profiles = {}

for name, hist in histograms.items():
    # 正規化
    avg_counts = hist / n_frames          # 平均每幀的原子數
    rho_z      = avg_counts / (A_xy * dz) # 數密度 (原子數/Å³)

    density_profiles[name] = rho_z

    # 寫入檔案
    filename = f"density_{name}.dat"
    np.savetxt(
        filename,
        np.column_stack([z_centers, rho_z]),
        header="z(Angstrom)  rho(atoms/Angstrom^3)",
        fmt="%.6f"
    )
    print(f"  ✅ 已儲存: {filename}")

print("\n=========================================")

# ============================================================
# --- 偵測電極邊界位置 (from start_nodrudes.pdb) ---
# ============================================================
print("🔍 正在偵測電極邊界...")
u_elec = Universe(electrode_pdb)
elecgrp = u_elec.select_atoms("resname grpc")
if len(elecgrp) == 0:
    raise ValueError(f"找不到 resname 'grpc' 的電極原子，請確認 {electrode_pdb}")
z_elec = elecgrp.positions[:, 2]
z_elec_left  = z_elec.min()   # 左側電極邊界 (Å)
z_elec_right = z_elec.max()   # 右側電極邊界 (Å)
print(f"   ✅ 左側電極邊界: {z_elec_left:.2f} Å")
print(f"   ✅ 右側電極邊界: {z_elec_right:.2f} Å\n")

# ============================================================
# --- 繪圖：2 張圖，TFSI 和 BMIM 分開 ---
#
# 版面配置：
#   上圖：TFSI（SO₂ vs CF₃）
#   下圖：BMIM（Im Ring vs CH₃ vs Butyl）
#
# 兩側都有電極，所以圖中會看到兩邊各有一個吸附層峰值。
# ============================================================
print("🎨 正在產生 1D Density Profile 圖表...")

plt.rcParams['font.family']       = 'sans-serif'
plt.rcParams['axes.linewidth']    = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.direction']   = 'in'
plt.rcParams['ytick.direction']   = 'in'

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle(
    '1D Density of Functional Groups under 1V',
    fontsize=18, fontweight='bold', y=0.97
)

plot_groups = {
    'TFSI': ['SO2_polar', 'CF3_nonpolar'],
    'BMIM': ['Im_polar', 'CH3_nonpolar', 'Butyl_nonpolar']
}

ion_titles = {
    'TFSI': 'TFSI Anion: Polar ($-SO_2$) vs Non-polar ($-CF_3$)',
    'BMIM': 'BMIM Cation: Polar (Im Ring) vs Non-polar ($-CH_3$, $-C_4H_9$)'
}

for ax, (ion_name, group_names) in zip(axes, plot_groups.items()):
    all_rho = []
    for name in group_names:
        filename = f"density_{name}.dat"
        if os.path.exists(filename):
            data  = np.loadtxt(filename, comments='#')
            z     = data[:, 0]
            rho_z = data[:, 1]
            all_rho.append(rho_z)
            info  = functional_groups[name]
            ax.plot(z, rho_z, lw=2.5,
                    color=info['color'],
                    linestyle=info['ls'],
                    label=info['label'])

    # --- 自動偵測有資料的 Z 範圍，消除右側空白 ---
    # 找出所有官能團密度加總後，密度 > 閾值的最大 z 位置
    if all_rho:
        total_rho  = np.sum(all_rho, axis=0)
        threshold  = total_rho.max() * 0.01   # 取最大密度的 1% 作為閾值
        nonzero_z  = z[total_rho > threshold]
        z_plot_min = max(0, nonzero_z.min() - 5)    # 左側多留 5 Å
        z_plot_max = nonzero_z.max() + 5             # 右側多留 5 Å
    else:
        z_plot_min, z_plot_max = 0, z_box

    ax.set_xlabel('Z position (Å)', fontsize=14)
    ax.set_ylabel('Number Density (atoms/Å$^3$)', fontsize=14)
    ax.set_title(ion_titles[ion_name], fontsize=14, fontweight='bold')
    ax.set_xlim(z_plot_min, z_plot_max)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=12, loc='upper left')

    # 電極位置精確標示（從 start_nodrudes.pdb 自動偵測）
    ax.axvline(x=z_elec_left,  color='green',  linestyle='--', linewidth=2,
               alpha=0.8, label=f'Positive electrode ({z_elec_left:.1f} Å)')
    ax.axvline(x=z_elec_right, color='orange', linestyle='--', linewidth=2,
               alpha=0.8, label=f'Negative electrode ({z_elec_right:.1f} Å)')
    ax.legend(fontsize=11, loc='upper left')

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
output_image = 'density_profile_1D_1V.png'
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"\n🎉 分析與繪圖完成！圖表已儲存為：{output_image}")