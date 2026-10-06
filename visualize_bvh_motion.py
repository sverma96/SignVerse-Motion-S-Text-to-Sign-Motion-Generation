import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from bvh import Bvh
import numpy as np


def visualize_motion(file_path):
    with open(file_path) as f:
        mocap = Bvh(f.read())

    frames = np.array([list(map(float, f)) for f in mocap.frames])

    # 🔥 Find most dynamic channels
    variance = np.var(frames, axis=0)
    top_indices = np.argsort(variance)[-20:]   # top 20 changing channels

    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'o-')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    def update(frame):
        vals = frames[frame][top_indices]

        # split into x and y
        half = len(vals) // 2
        x = vals[:half]
        y = vals[half:half*2]

        # normalize (so movement is visible)
        x = (x - x.mean()) / (x.std() + 1e-6)
        y = (y - y.mean()) / (y.std() + 1e-6)

        line.set_data(x, y)
        return line,

    ani = FuncAnimation(fig, update, frames=min(200, len(frames)), interval=50)

    plt.title("BVH Motion (Auto-Selected Dynamic Channels)")
    plt.xlabel("Dynamic X")
    plt.ylabel("Dynamic Y")

    plt.show()


if __name__ == "__main__":
    file_path = "C:/Users/HP/Downloads/Motion-S-Text-to-Sign-Motion-Generation-Signvrse-main/motion-s-hierarchical-text-to-motion-generation-for-sign-language/Train/1433/1433.bvh"
    visualize_motion(file_path)