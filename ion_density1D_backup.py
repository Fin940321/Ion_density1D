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
print(len(u.trajectory))
print(frameStart)
print(frameNeed)

avgfreq=1 # (自己設定)
frameAverage = int(frameNeed / avgfreq)

#輸出結果來確認:
print(frameAverage)

bins = 1000 # (自己設定)
dz = u.trajectory[0].dimensions[2] / bins
cation = u.select_atoms("resname BMI")
anion  = u.select_atoms("resname trf")

#輸出結果來確認:
print(f"Box Z dimension: {u.trajectory[0].dimensions[2]:.2f} Å")
print(f"Bin width (dz): {dz:.4f} Å")
print(f"Number of cations: {len(cation)}")
print(f"Number of anions: {len(anion)}")
print()

# print("陽離子的位置:\n", cation.positions)
# print("陰離子的位置:\n", anion.positions)

# 使用 numba 優化的計算函數
@jit(nopython=True)
def calculate_density_counts(positions_z, dz, bins):
    """使用 numba 加速的密度計算函數"""
    counts = np.zeros(bins, dtype=np.int64)
    for z in positions_z:
        bin_idx = int(z // dz)
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
    
    # 使用優化後的函數計算
    cation_counts += calculate_density_counts(cation.positions[:, 2], dz, bins)
    anion_counts += calculate_density_counts(anion.positions[:, 2], dz, bins)

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
# matplotlib 繪製離子密度分布圖 (合併版)
z_positions = np.linspace(0, u.trajectory[0].dimensions[2], bins)

# 創建組合圖表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=300)

# 陽離子密度圖
ax1.plot(z_positions, cation_Density, label="Cation (BMI)", color='blue', linewidth=1.5)
ax1.set_title("Cation Density Distribution", fontsize=14, fontweight='bold')
ax1.set_xlabel("Z Position (Å)", fontsize=12)
ax1.set_ylabel("Normalized Density", fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(10, 98)

# 陰離子密度圖
ax2.plot(z_positions, anion_Density, label="Anion (TRF)", color='red', linewidth=1.5)
ax2.set_title("Anion Density Distribution", fontsize=14, fontweight='bold')
ax2.set_xlabel("Z Position (Å)", fontsize=12)
ax2.set_ylabel("Normalized Density", fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(10, 98)

plt.tight_layout()
plt.savefig("Ion_Density_Distribution_Combined.png", dpi=600, bbox_inches='tight')
print("已儲存合併圖表: Ion_Density_Distribution_Combined.png")
# plt.show()
plt.close()
# ----------------------------------------------------
# 繪製陰陽離子對比圖 (單一圖表)
plt.figure(figsize=(12, 6), dpi=300)

plt.plot(z_positions, cation_Density, label="Cation (BMI)", color='blue', linewidth=1.5, alpha=0.8)
plt.plot(z_positions, anion_Density, label="Anion (TRF)", color='red', linewidth=1.5, alpha=0.8)

plt.title("Ion Density Distribution Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Z Position (Å)", fontsize=12)
plt.ylabel("Normalized Density", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(10, 98)

plt.tight_layout()
plt.savefig("Ion_Density_Comparison.png", dpi=600, bbox_inches='tight')
print("已儲存對比圖表: Ion_Density_Comparison.png")
# plt.show()
plt.close()
print("\n所有圖表已生成完畢！")