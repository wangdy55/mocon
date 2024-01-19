from scene.Scene import Scene
from scene.Prototype import Prototype

def main():
    scene = Scene()
    proto = Prototype(scene)
    
    scene.run()

if __name__ == '__main__':
    main()