"""
Ion Density Distribution Analysis - V3 Linus Edition

"Good taste means eliminating special cases and doing things right the first time."
                                                        - Linus Torvalds
"""

import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
import time
import sys


# ============================================================
# Configuration Parameters
# ============================================================
class Config:
    """Centralized configuration - no magic numbers scattered around"""
    # File paths
    TOPOLOGY = "start_drudes.pdb"
    TRAJECTORY = "FV_NVT.dcd"
    
    # Analysis parameters
    NUM_BINS = 1000
    FRAME_SKIP = 1  # Analyze every N frames (1 = all frames)
    
    # Ion types to detect (priority order)
    CATION_TYPES = ["BMIM", "BMI", "EMIM", "PMIM", "OMIM"]
    ANION_TYPES = ["Tf2N", "trfl", "Tf2", "TRF", "trf", "TFSI", "BF4", "PF6", "Cl"]
    
    # Electrode residue name
    ELECTRODE_RESNAME = "grp"
    
    # Output settings
    PLOT_DPI = 600


# ============================================================
# Core computation functions
# ============================================================
@jit(nopython=True)
def compute_density_histogram(positions_z, z_min, z_max, bins):
    """
    Numba-accelerated histogram computation.
    
    Good taste: Simple, fast, no branches in the loop.
    """
    counts = np.zeros(bins, dtype=np.float64)  # Use float from the start!
    dz = (z_max - z_min) / bins
    
    for z in positions_z:
        if z_min <= z <= z_max:
            bin_idx = int((z - z_min) / dz)
            if 0 <= bin_idx < bins:
                counts[bin_idx] += 1.0
    
    return counts


def detect_ion_type(universe, ion_types, ion_category):
    """
    Auto-detect ion type from topology.
    
    Returns: (AtomGroup, residue_name)
    Raises: ValueError if not found
    """
    for ion_name in ion_types:
        ions = universe.select_atoms(f"resname {ion_name}")
        if len(ions) > 0:
            return ions, ion_name
    
    raise ValueError(
        f"Cannot detect {ion_category}! "
        f"Tried: {ion_types}\n"
        f"Available residues: {set(res.name for res in universe.residues)}"
    )


def calculate_ion_densities(universe, cation, anion, z_min, z_max, 
                           num_bins, frame_start, frame_count, frame_skip):
    """
    Calculate ion number densities along Z-axis.
    
    Returns: (z_positions, cation_density, anion_density)
    Units: number density in Å⁻¹
    """
    # Pre-allocate accumulator arrays (float64, not int64!)
    cation_hist = np.zeros(num_bins, dtype=np.float64)
    anion_hist = np.zeros(num_bins, dtype=np.float64)
    
    frames_analyzed = 0
    
    print("\nComputing ion density distributions...")
    start_time = time.time()
    
    for i in tqdm(range(frame_count), desc="Processing", unit="frame", ncols=70):
        frame_idx = frame_start + i * frame_skip
        
        if frame_idx >= len(universe.trajectory):
            break
        
        universe.trajectory[frame_idx]
        
        # Accumulate counts
        cation_hist += compute_density_histogram(
            cation.positions[:, 2], z_min, z_max, num_bins
        )
        anion_hist += compute_density_histogram(
            anion.positions[:, 2], z_min, z_max, num_bins
        )
        
        frames_analyzed += 1
    
    elapsed = time.time() - start_time
    print(f"\n✓ Analysis complete")
    print(f"  Frames analyzed: {frames_analyzed}")
    print(f"  Total time:      {elapsed:.2f} s")
    print(f"  Per frame:       {elapsed/frames_analyzed*1000:.2f} ms")
    print()
    
    # Calculate bin properties
    dz = (z_max - z_min) / num_bins
    z_positions = np.linspace(z_min + dz/2, z_max - dz/2, num_bins)
    
    # Normalize to number density (particles per Å)
    # Formula: density = counts / (num_particles * num_frames * bin_width)
    cation_density = cation_hist / (len(cation) * frames_analyzed * dz)
    anion_density = anion_hist / (len(anion) * frames_analyzed * dz)
    
    # Validate normalization (integral should be ~1)
    cation_integral = np.trapz(cation_density, z_positions)
    anion_integral = np.trapz(anion_density, z_positions)
    
    print("=== Density Statistics ===")
    print(f"  Cation density integral: {cation_integral:.6f} (should be ~1)")
    print(f"  Anion density integral:  {anion_integral:.6f} (should be ~1)")
    print(f"  Bin width (dz):          {dz:.4f} Å")
    print()
    
    # Sanity checks
    if not (0.8 <= cation_integral <= 1.2):
        print(f"⚠ WARNING: Cation normalization looks wrong ({cation_integral:.3f})")
    if not (0.8 <= anion_integral <= 1.2):
        print(f"⚠ WARNING: Anion normalization looks wrong ({anion_integral:.3f})")
    
    return z_positions, cation_density, anion_density


def save_plot(fig, filename, config):
    """Save plot with consistent settings"""
    fig.savefig(filename, dpi=config.PLOT_DPI, bbox_inches='tight')
    print(f"  ✓ {filename}")
    plt.close(fig)


def plot_density_single(z_pos, density, label, color, title, filename, config, z_min=None, z_max=None):
    """Plot single ion density distribution"""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    ax.plot(z_pos, density, label=label, color=color, linewidth=1.5)
    
    # 標示電極位置
    if z_min is not None and z_max is not None:
        ax.axvline(x=z_min, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Cathode ({z_min:.1f} Å)')
        ax.axvline(x=z_max, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Anode ({z_max:.1f} Å)')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Z Position (Å)", fontsize=12)
    ax.set_ylabel("Number Density (Å⁻¹)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_plot(fig, filename, config)


def plot_density_combined(z_pos, cat_dens, an_dens, cat_name, an_name, config, z_min=None, z_max=None):
    """Plot cation and anion densities together"""
    # Subplot version
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=300)
    
    ax1.plot(z_pos, cat_dens, label=f'Cation ({cat_name})', 
             color='blue', linewidth=1.5)
    # 標示電極位置
    if z_min is not None and z_max is not None:
        ax1.axvline(x=z_min, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Cathode ({z_min:.1f} Å)')
        ax1.axvline(x=z_max, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Anode ({z_max:.1f} Å)')
    ax1.set_title(f"Cation Density Distribution of {cat_name}", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Z Position (Å)", fontsize=12)
    ax1.set_ylabel("Number Density (Å⁻¹)", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(z_pos, an_dens, label=f'Anion ({an_name})', 
             color='red', linewidth=1.5)
    # 標示電極位置
    if z_min is not None and z_max is not None:
        ax2.axvline(x=z_min, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Cathode ({z_min:.1f} Å)')
        ax2.axvline(x=z_max, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Anode ({z_max:.1f} Å)')
    ax2.set_title(f"Anion Density Distribution of {an_name}", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Z Position (Å)", fontsize=12)
    ax2.set_ylabel("Number Density (Å⁻¹)", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_plot(fig, "Ion_Density_Combined_2V.png", config)
    
    # Overlay version
    fig2, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    ax.plot(z_pos, cat_dens, label=f'Cation ({cat_name})', 
            color='blue', linewidth=1.5, alpha=0.8)
    ax.plot(z_pos, an_dens, label=f'Anion ({an_name})', 
            color='red', linewidth=1.5, alpha=0.8)
    # 標示電極位置
    if z_min is not None and z_max is not None:
        ax.axvline(x=z_min, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Cathode ({z_min:.1f} Å)')
        ax.axvline(x=z_max, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Anode ({z_max:.1f} Å)')
    ax.set_title(f"Ion Density Comparison under 2V", fontsize=14, fontweight='bold')
    ax.set_xlabel("Z Position (Å)", fontsize=12)
    ax.set_ylabel("Number Density (Å⁻¹)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    fig2.tight_layout()
    save_plot(fig2, "Ion_Density_Comparison_2V.png", config)


# ============================================================
# Main analysis pipeline
# ============================================================
def main():
    config = Config()
    
    print("=" * 70)
    print("  Ion Density Distribution Analysis - V3 Linus Edition")
    print("=" * 70)
    print()
    
    # Load trajectory
    print("=== Loading Trajectory ===")
    print(f"  Topology:   {config.TOPOLOGY}")
    print(f"  Trajectory: {config.TRAJECTORY}")
    
    u = mda.Universe(config.TOPOLOGY, config.TRAJECTORY)
    
    total_frames = len(u.trajectory)
    frame_start = total_frames // 2  # Start from equilibrated region
    frame_count = (total_frames - frame_start) // config.FRAME_SKIP
    
    print(f"✓ Loaded successfully")
    print(f"  Total frames:     {total_frames}")
    print(f"  Analysis start:   {frame_start}")
    print(f"  Frames to analyze: {frame_count}")
    print(f"  Frame skip:       {config.FRAME_SKIP}")
    print()
    
    # Detect electrode boundaries
    print("=== Detecting Electrode Boundaries ===")
    electrode = u.select_atoms(f"resname {config.ELECTRODE_RESNAME}")
    
    if len(electrode) == 0:
        raise ValueError(f"No electrode atoms found with resname '{config.ELECTRODE_RESNAME}'")
    
    z_positions_electrode = electrode.positions[:, 2]
    z_min = z_positions_electrode.min()
    z_max = z_positions_electrode.max()
    z_range = z_max - z_min
    
    print(f"  Electrode resname: {config.ELECTRODE_RESNAME}")
    print(f"  Number of atoms:   {len(electrode)}")
    print(f"  Z range:           {z_min:.2f} to {z_max:.2f} Å")
    print(f"  Spacing:           {z_range:.2f} Å")
    print()
    
    # Auto-detect ions
    print("=== Detecting Ion Types ===")
    cation, cat_name = detect_ion_type(u, config.CATION_TYPES, "cation")
    anion, an_name = detect_ion_type(u, config.ANION_TYPES, "anion")
    
    print(f"  ✓ Cation: {cat_name} ({len(cation)} atoms)")
    print(f"  ✓ Anion:  {an_name} ({len(anion)} atoms)")
    print()
    
    # Validate
    assert len(cation) > 0, "No cation atoms found!"
    assert len(anion) > 0, "No anion atoms found!"
    
    # Calculate densities
    print(f"=== Analysis Parameters ===")
    print(f"  Number of bins: {config.NUM_BINS}")
    print(f"  Bin width:      {z_range/config.NUM_BINS:.4f} Å")
    
    z_pos, cat_dens, an_dens = calculate_ion_densities(
        u, cation, anion, z_min, z_max, 
        config.NUM_BINS, frame_start, frame_count, config.FRAME_SKIP
    )
    
    # Generate plots
    print("=== Generating Plots ===")
    
    plot_density_single(
        z_pos, cat_dens, 
        f'Cation ({cat_name})', 'blue',
        "Cation Density Distribution",
        "Cation_Density_2V.png", config,
        z_min=z_min, z_max=z_max
    )
    
    plot_density_single(
        z_pos, an_dens,
        f'Anion ({an_name})', 'red', 
        "Anion Density Distribution",
        "Anion_Density_2V.png", config,
        z_min=z_min, z_max=z_max
    )
    
    plot_density_combined(z_pos, cat_dens, an_dens, cat_name, an_name, config, z_min=z_min, z_max=z_max)
    
    print()
    print("=" * 70)
    print("✓ Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
