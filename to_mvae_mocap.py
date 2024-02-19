import os

from mocon.motion.mocap.BVHMotion import BVHMotion

bvh_dir = "mocon/motion/mocap/bvh"
npz_dir = "mocon/motion/mocap/npz"
bvh_file = os.path.join(bvh_dir, "run2_subject1.bvh")
npz_file = os.path.join(npz_dir, "run2_subject1.npz")

def main():
    bvh = BVHMotion(bvh_file)
    bvh.save_mvae_mocap(npz_file)
    print("Done!")

if __name__ == '__main__':
    main()