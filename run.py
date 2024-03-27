from scene.Scene import Scene
from scene.Model import Model
from mocon.Mocon import Mocon

def main():
    scene = Scene()
    model = Model(scene)

    Mocon(
        scene, model,
        mode="ctrl"
    )
    
    scene.run()

if __name__ == '__main__':
    main()