import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
import time

u = mda.Universe("start_drudes.pdb", "FV_NVT.dcd")
u.guess_TopologyAttrs(context='default', to_guess=['elements'])

frameStart = int(len(u.trajectory) / 2) # (抓總幀數的一半開始分析)
#frameStart = 0
frameNeed = len(u.trajectory) - frameStart

#輸出結果來確認:
print("\nTotal frame:", len(u.trajectory))
print("Frame Start:", frameStart)
print("Frame Need:", frameNeed)

avgfreq=1 # (自己設定)
frameAverage = int(frameNeed / avgfreq)

#輸出結果來確認:
print("Frame Average:", frameAverage)

# 選擇 grp 原子並獲取 z 軸範圍
grp_atoms = u.select_atoms("resname grp")
grp_z_positions = grp_atoms.positions[:, 2]  # 獲取所有 grp 原子的 z 座標

# 找到第一個和最後一個 grp 的 z 軸座標
z_min = grp_z_positions.min()  # 最小 z 座標（起始點）
z_max = grp_z_positions.max()  # 最大 z 座標（終點）
z_range = z_max - z_min  # z 軸範圍

print(f"\n=== 電極 (grp) Z 軸範圍 ===")
print(f"第一個電極 (起點) z 座標: {z_min:.2f} Å")
print(f"第二個電極 (終點) z 座標: {z_max:.2f} Å")
print(f"電極間距離: {z_range:.2f} Å")
print(f"原始盒子 Z 維度: {u.trajectory[0].dimensions[2]:.2f} Å")

bins = 1000 # (自己設定)
dz = z_range / bins  # 使用電極間距離而非整個盒子

# ============================================================
# 自動偵測陽離子和陰離子類型
# ============================================================
# 嘗試常見的陽離子類型
cation_types = ["BMI", "BMIM", "EMIM", "PMIM"]
cation = None
cation_name = None

for cat_type in cation_types:
    temp_cation = u.select_atoms(f"resname {cat_type}")
    if len(temp_cation) > 0:
        cation = temp_cation
        cation_name = cat_type
        break

# 嘗試常見的陰離子類型
anion_types = ["Tf2", "TRF", "trf", "TFSI", "BF4", "PF6", "Cl"]
anion = None
anion_name = None

for an_type in anion_types:
    temp_anion = u.select_atoms(f"resname {an_type}")
    if len(temp_anion) > 0:
        anion = temp_anion
        anion_name = an_type
        break

# 檢查是否成功偵測到離子
if cation is None or len(cation) == 0:
    raise ValueError("無法偵測到陽離子！請檢查拓撲文件中的殘基名稱。")
if anion is None or len(anion) == 0:
    raise ValueError("無法偵測到陰離子！請檢查拓撲文件中的殘基名稱。")

#輸出結果來確認:
print(f"\n=== 離子偵測結果 ===")
print(f"偵測到的陽離子類型: {cation_name}")
print(f"偵測到的陰離子類型: {anion_name}")
print(f"\n=== 分析參數 ===")
print(f"分析區間: {z_min:.2f} Å 到 {z_max:.2f} Å")
print(f"Bin 數量: {bins}")
print(f"Bin 寬度 (dz): {dz:.4f} Å")
print(f"陽離子數量 ({cation_name}): {len(cation)}")
print(f"陰離子數量 ({anion_name}): {len(anion)}")
print(f"電極原子數量 (grp): {len(grp_atoms)}")
print()

# print("陽離子的位置:\n", cation.positions)
# print("陰離子的位置:\n", anion.positions)

# 使用 numba 優化的計算函數
@jit(nopython=True)
def calculate_density_counts(positions_z, z_min, z_range, dz, bins):
    """使用 numba 加速的密度計算函數，基於電極間的 z 軸範圍"""
    counts = np.zeros(bins, dtype=np.int64)
    for z in positions_z:
        # 將 z 座標轉換為相對於 z_min 的位置
        z_relative = z - z_min
        if 0 <= z_relative <= z_range:  # 只統計在電極之間的離子
            bin_idx = int(z_relative / dz)
            if 0 <= bin_idx < bins:  # 確保索引在範圍內
                counts[bin_idx] += 1
    return counts

# 初始化為 numpy 陣列
cation_counts = np.zeros(bins, dtype=np.int64)
anion_counts = np.zeros(bins, dtype=np.int64)

# 開始計時
start_time = time.time()

for i in tqdm(range(frameAverage), desc="計算離子密度", unit="frame"):
    frameCurrent = frameStart + i * avgfreq
    if frameCurrent >= len(u.trajectory):
        break
        
    u.trajectory[frameCurrent]
    
    # 使用優化後的函數計算（基於電極間的 z 軸範圍）
    cation_counts += calculate_density_counts(cation.positions[:, 2], z_min, z_range, dz, bins)
    anion_counts += calculate_density_counts(anion.positions[:, 2], z_min, z_range, dz, bins)

# 計算執行時間
elapsed_time = time.time() - start_time
print(f"\n計算完成！耗時: {elapsed_time:.2f} 秒")
print(f"平均每幀處理時間: {elapsed_time/frameAverage*1000:.2f} 毫秒")
print()

# 歸一化 (簡化版，一次完成)
cation_Density = cation_counts / len(cation) / frameAverage / bins
anion_Density = anion_counts / len(anion) / frameAverage / bins

print(f"Cation density sum: {cation_Density.sum():.6f}")
print(f"Anion density sum: {anion_Density.sum():.6f}")
print()


# ----------------------------------------------------
# matplotlib 繪製離子密度分布圖
# 使用電極間的 z 軸範圍
z_positions = np.linspace(z_min, z_max, bins)

# 動態生成圖例標籤
cation_label = f"Cation ({cation_name})"
anion_label = f"Anion ({anion_name})"

# 單獨繪製陽離子密度圖
plt.figure(figsize=(12, 6), dpi=300)
plt.plot(z_positions, cation_Density, label=cation_label, color='blue', linewidth=1.5)
plt.title("Cation Density Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Z Position (Å)", fontsize=12)
plt.ylabel("Normalized Density (Å⁻¹)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(z_min, z_max)  # 使用電極範圍
plt.tight_layout()
plt.savefig("Cation_Density_Distribution.png", dpi=600, bbox_inches='tight')
print("已儲存陽離子密度圖: Cation_Density_Distribution.png")
plt.close()

# 單獨繪製陰離子密度圖
plt.figure(figsize=(12, 6), dpi=300)
plt.plot(z_positions, anion_Density, label=anion_label, color='red', linewidth=1.5)
plt.title("Anion Density Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Z Position (Å)", fontsize=12)
plt.ylabel("Normalized Density (Å⁻¹)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(z_min, z_max)  # 使用電極範圍
plt.tight_layout()
plt.savefig("Anion_Density_Distribution.png", dpi=600, bbox_inches='tight')
print("已儲存陰離子密度圖: Anion_Density_Distribution.png")
plt.close()

# 創建組合圖表 (上下排列)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=300)

# 陽離子密度圖
ax1.plot(z_positions, cation_Density, label=cation_label, color='blue', linewidth=1.5)
ax1.set_title("Cation Density Distribution", fontsize=14, fontweight='bold')
ax1.set_xlabel("Z Position (Å)", fontsize=12)
ax1.set_ylabel("Normalized Density (Å⁻¹)", fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(z_min, z_max)  # 使用電極範圍

# 陰離子密度圖
ax2.plot(z_positions, anion_Density, label=anion_label, color='red', linewidth=1.5)
ax2.set_title("Anion Density Distribution", fontsize=14, fontweight='bold')
ax2.set_xlabel("Z Position (Å)", fontsize=12)
ax2.set_ylabel("Normalized Density (Å⁻¹)", fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(z_min, z_max)  # 使用電極範圍

plt.tight_layout()
plt.savefig("Ion_Density_Distribution_Combined.png", dpi=600, bbox_inches='tight')
print("已儲存合併圖表: Ion_Density_Distribution_Combined.png")
plt.close()
# ----------------------------------------------------
# 繪製陰陽離子對比圖 (單一圖表)
plt.figure(figsize=(12, 6), dpi=300)

plt.plot(z_positions, cation_Density, label=cation_label, color='blue', linewidth=1.5, alpha=0.8)
plt.plot(z_positions, anion_Density, label=anion_label, color='red', linewidth=1.5, alpha=0.8)

plt.title("Ion Density Distribution Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Z Position (Å)", fontsize=12)
plt.ylabel("Normalized Density (Å⁻¹)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(z_min, z_max)  # 使用電極範圍

plt.tight_layout()
plt.savefig("Ion_Density_Comparison.png", dpi=600, bbox_inches='tight')
print("已儲存對比圖表: Ion_Density_Comparison.png")
# plt.show()
plt.close()
print("\n所有圖表已生成完畢！")
