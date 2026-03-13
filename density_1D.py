from MDAnalysis import *
import numpy as np
import matplotlib.pyplot as plt

u = Universe("start_drudes.pdb", "FV_NVT.dcd")

startFrame = 1000
frameCount = u.trajectory.n_frames - startFrame
# this is in case of a long trajectory, for this 
# trajectory we can average every couple timesteps
avgfreq=2
nFramesToAverage = int(frameCount / avgfreq)

# note that u.trajectory[0].dimensions[2] is much
# bigger than actual system because it includes the vaccuum gap
# so use 2-3* more bins than the resolution we want

bins = 260
dz = u.trajectory[0].dimensions[2] / bins
cation = u.select_atoms("resname BMI and name C1")
anion  = u.select_atoms("resname Tf2N and name Cf")

cat_counts = [0 for y in range(bins)]
an_counts  = [0 for y in range(bins)]

for i in range(nFramesToAverage):
    currentFrame = startFrame + i * avgfreq
    if currentFrame >= u.trajectory.n_frames:
        break
    u.trajectory[currentFrame]
    for atom in cation.positions:
        cat_counts[int(atom[2] // dz)] += 1
    for atom in anion.positions:
        an_counts[int(atom[2] // dz)] += 1

cat_counts=np.asfarray(cat_counts)
an_counts=np.asfarray(an_counts)

# normalize
cat_counts = cat_counts / len(cation) / nFramesToAverage / bins
an_counts = an_counts / len(anion) / nFramesToAverage / bins

#print("\ncation:")
#for i in range(len(cat_counts)):
    #print( i , cat_counts[i])
#print("\nanion:")
#for i in range(len(an_counts)):
    #print( i , an_counts[i])

#matplotlib
# 設定 z 軸位置
z_positions = np.linspace(0, u.trajectory[0].dimensions[2], bins)

# 繪製圖表
plt.figure(figsize=(10, 6))

# 正離子分布
plt.plot(z_positions, cat_counts, label="Cation (BMI)", color='blue')

# 負離子分布
plt.plot(z_positions, an_counts, label="Anion (TRF)", color='red')

# 添加圖表標籤
plt.title("Cation and Anion Density Distribution")
plt.xlabel("Ion Position (Å)")
plt.ylabel("Normalized Density")
plt.legend()
plt.grid()
plt.xlim(0, 100)

# 顯示圖表
plt.show()
