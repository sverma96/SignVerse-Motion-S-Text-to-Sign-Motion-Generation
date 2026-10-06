import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from bvh import Bvh


# ---------------- ROTATIONS ----------------
def rot_x(a):
    r = np.radians(a)
    return np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])

def rot_y(a):
    r = np.radians(a)
    return np.array([[np.cos(r),0,np.sin(r)],[0,1,0],[-np.sin(r),0,np.cos(r)]])

def rot_z(a):
    r = np.radians(a)
    return np.array([[np.cos(r),-np.sin(r),0],[np.sin(r),np.cos(r),0],[0,0,1]])

def R(rx, ry, rz):
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


# ---------------- MAIN ----------------
def visualize_fk_skeleton_3d(file_path):
    with open(file_path) as f:
        mocap = Bvh(f.read())

    joints = mocap.get_joints_names()
    frames = mocap.frames

    # Parents (safe extraction)
    parents = {}
    for j in joints:
        p = mocap.joint_parent(j)
        parents[j] = p.name if p else None

    # Offsets
    offsets = {}
    for j in joints:
        try:
            offsets[j] = np.array(list(map(float, mocap.joint_offset(j))))
        except:
            offsets[j] = np.zeros(3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Clean white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    def update(frame_idx):
        ax.cla()

        frame = list(map(float, frames[frame_idx]))

        pos = {}
        rot = {}
        idx = 0

        for j in joints:
            parent = parents[j]

            if parent is None:
                p = np.array(frame[idx:idx+3])
                idx += 3

                rx, ry, rz = frame[idx:idx+3]
                idx += 3

                rot[j] = R(rx, ry, rz)
                pos[j] = p

            else:
                if parent not in pos:
                    continue

                rx, ry, rz = frame[idx:idx+3]
                idx += 3

                r_local = R(rx, ry, rz)

                pos[j] = pos[parent] + rot[parent] @ offsets[j]
                rot[j] = rot[parent] @ r_local

        # 🔥 Center skeleton (straight)
        all_pts = np.array(list(pos.values()))
        center = all_pts.mean(axis=0)
        for k in pos:
            pos[k] = pos[k] - center

        # 🔴 Plot joints (axis swap for upright view)
        for p in pos.values():
            ax.scatter(p[0], p[2], p[1], c='red', s=25)

        # ⚫ Draw bones
        for j in joints:
            parent = parents[j]
            if parent and parent in pos and j in pos:
                x = [pos[parent][0], pos[j][0]]
                y = [pos[parent][2], pos[j][2]]
                z = [pos[parent][1], pos[j][1]]
                ax.plot(x, y, z, 'k-', linewidth=2)

        # 🔥 Tight limits (reduces box feeling)
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(-50, 50)

        # 🔥 REMOVE EVERYTHING VISUAL
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)

        # Remove panes
        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)

        # Remove pane edges
        ax.xaxis.pane.set_edgecolor((1,1,1,0))
        ax.yaxis.pane.set_edgecolor((1,1,1,0))
        ax.zaxis.pane.set_edgecolor((1,1,1,0))

        # 🔥 REMOVE AXIS LINES (fix black corner issue)
        ax.xaxis.line.set_color((1,1,1,0))
        ax.yaxis.line.set_color((1,1,1,0))
        ax.zaxis.line.set_color((1,1,1,0))

        ax.set_title("")

    ani = FuncAnimation(fig, update, frames=min(200, len(frames)), interval=50)
    plt.show()


# ---------------- RUN ----------------
if __name__ == "__main__":
    file_path = "C:/Users/HP/Downloads/Motion-S-Text-to-Sign-Motion-Generation-Signvrse-main/motion-s-hierarchical-text-to-motion-generation-for-sign-language/Train/1449/1449.bvh"
    visualize_fk_skeleton_3d(file_path)