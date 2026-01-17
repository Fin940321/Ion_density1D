import mdtraj as md
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument("--output", type=str, help="PDB filename from MD trajectory")
#args = parser.parse_args()

t = md.load("FV_NVT.dcd",top="start_drudes.pdb")
#t[-1].save_pdb(str(args.output))
t[-600].save_pdb("3V_2cnt_otfbmim_start.pdb")
