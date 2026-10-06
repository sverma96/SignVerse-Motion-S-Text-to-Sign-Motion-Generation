import matplotlib.pyplot as plt
from bvh import Bvh


def visualize_skeleton(file_path):
    with open(file_path) as f:
        mocap = Bvh(f.read())

    joints = mocap.get_joints_names()

    joint_positions = {}

    for joint in joints:
        try:
            offset = mocap.joint_offset(joint)
            x, y, z = map(float, offset)
            joint_positions[joint] = (x, z)
        except:
            continue

    fig, ax = plt.subplots()

    # plot joints
    for joint, (x, y) in joint_positions.items():
        ax.scatter(x, y, color='red')

    # 🔥 safe parent-child connections
    for joint in joints:
        parent = mocap.joint_parent(joint)

        if parent and parent in joint_positions and joint in joint_positions:
            x1, y1 = joint_positions[parent]
            x2, y2 = joint_positions[joint]
            ax.plot([x1, x2], [y1, y2], 'k-')

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_title("BVH Skeleton (Safe Hierarchy)")

    plt.show()


if __name__ == "__main__":
    file_path = "C:/Users/HP/Downloads/Motion-S-Text-to-Sign-Motion-Generation-Signvrse-main/motion-s-hierarchical-text-to-motion-generation-for-sign-language/Train/1433/1433.bvh"
    visualize_skeleton(file_path)