import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from bvh import Bvh


# 🔥 Rotation matrices
def rot_x(angle):
    rad = np.radians(angle)
    return np.array([
        [1, 0, 0],
        [0, np.cos(rad), -np.sin(rad)],
        [0, np.sin(rad),  np.cos(rad)]
    ])

def rot_y(angle):
    rad = np.radians(angle)
    return np.array([
        [ np.cos(rad), 0, np.sin(rad)],
        [0,            1, 0],
        [-np.sin(rad), 0, np.cos(rad)]
    ])

def rot_z(angle):
    rad = np.radians(angle)
    return np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad),  np.cos(rad), 0],
        [0,            0,           1]
    ])


def get_rotation_matrix(rx, ry, rz):
    # ZYX order (common in BVH)
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def visualize_fk_skeleton(file_path):
    with open(file_path) as f:
        mocap = Bvh(f.read())

    joints = mocap.get_joints_names()
    frames = mocap.frames

    # 🔥 Build parent mapping (convert BvhNode → string name safely)
    parents = {}
    for joint in joints:
        parent_node = mocap.joint_parent(joint)

        if parent_node is None:
            parent_name = None
        else:
            # Extract string name safely
            try:
                parent_name = parent_node.name
            except AttributeError:
                parent_name = str(parent_node)

            # Clean ROOT prefix if present
            parent_name = parent_name.replace("ROOT ", "").strip()

        parents[joint] = parent_name

    # 🔥 Offsets
    offsets = {}
    for joint in joints:
        try:
            offsets[joint] = np.array(list(map(float, mocap.joint_offset(joint))))
        except:
            offsets[joint] = np.zeros(3)

    fig, ax = plt.subplots()

    def update(frame_idx):
        ax.clear()

        frame = list(map(float, frames[frame_idx]))

        joint_positions = {}
        joint_rotations = {}

        idx = 0

        for joint in joints:
            parent = parents[joint]

            # 🔥 ROOT joint
            if parent is None:
                pos = np.array(frame[idx:idx+3])
                idx += 3

                rx, ry, rz = frame[idx:idx+3]
                idx += 3

                R = get_rotation_matrix(rx, ry, rz)

                joint_positions[joint] = pos
                joint_rotations[joint] = R

            else:
                # Skip if parent not yet computed
                if parent not in joint_positions:
                    continue

                rx, ry, rz = frame[idx:idx+3]
                idx += 3

                R_local = get_rotation_matrix(rx, ry, rz)

                parent_pos = joint_positions[parent]
                parent_rot = joint_rotations[parent]

                # 🔥 Forward Kinematics
                global_pos = parent_pos + parent_rot @ offsets[joint]
                global_rot = parent_rot @ R_local

                joint_positions[joint] = global_pos
                joint_rotations[joint] = global_rot

        # 🔥 Plot joints
        for pos in joint_positions.values():
            x, y, z = pos
            ax.scatter(x, z, color='red')

        # 🔥 Draw bones
        for joint in joints:
            parent = parents[joint]
            if parent and parent in joint_positions and joint in joint_positions:
                x1, y1, z1 = joint_positions[parent]
                x2, y2, z2 = joint_positions[joint]
                ax.plot([x1, x2], [z1, z2], 'k-')

        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_title("FK Skeleton Animation")

    ani = FuncAnimation(fig, update, frames=min(200, len(frames)), interval=50)
    plt.show()


# 🔥 MAIN
if __name__ == "__main__":
    file_path = "C:/Users/HP/Downloads/Motion-S-Text-to-Sign-Motion-Generation-Signvrse-main/motion-s-hierarchical-text-to-motion-generation-for-sign-language/Train/1433/1433.bvh"
    visualize_fk_skeleton(file_path)