from scene.Scene import Scene
from scene.Model import Model
from mocon.Mocon import Mocon

bvh_path = "mocon/motion/mocap/bvh/walk1_subject5.bvh"
npz_path = "mocon/motion/mocap/npz/walk1_subject5.npz"
mvae_path = "mocon/motion/mvae/model/walk1_subject5_240129_190220.pt"

def main():
    scene = Scene()
    model = Model(scene)

    Mocon(scene, model)
    
    scene.run()

if __name__ == '__main__':
    main()