from bvh import Bvh
import os


# -------------------------------
# Load BVH file
# -------------------------------
def load_bvh(file_path):
    with open(file_path) as f:
        mocap = Bvh(f.read())
    return mocap


# -------------------------------
# Explore single BVH file
# -------------------------------
def explore_bvh(file_path):
    mocap = load_bvh(file_path)

    print("\n==============================")
    print(f"📂 File: {file_path}")
    print("==============================")

    # Basic info
    print("Total Frames:", mocap.nframes)
    print("Frame Time:", mocap.frame_time)

    # Joint information
    joints = mocap.get_joints_names()
    print("\nTotal Joints:", len(joints))
    print("First 10 Joints:", joints[:10])

    # Hierarchy (parent-child)
    print("\n--- Joint Hierarchy (first 5) ---")
    for joint in joints[:5]:
        parent = mocap.joint_parent(joint)
        print(f"{joint} → Parent: {parent}")

    # Sample frame data
    print("\n--- Sample Frame Data ---")
    print("Frame 0:", mocap.frames[0][:10])
    print("Frame 1:", mocap.frames[1][:10])

    print("\nNOTE:")
    print("✔ Each frame = one body pose")
    print("✔ Values = joint rotations/positions")
    print("✔ Frames sequence = animation")


# -------------------------------
# Print multiple frames
# -------------------------------
def print_multiple_frames(file_path, num_frames=3):
    mocap = load_bvh(file_path)

    print(f"\n--- Showing First {num_frames} Frames ---")

    for i in range(num_frames):
        print(f"\nFrame {i}:")
        print(mocap.frames[i][:10])


# -------------------------------
# Explore multiple BVH files (dataset)
# -------------------------------
def explore_multiple_bvh(train_folder):
    print("\n🔍 Exploring Multiple BVH Files from Dataset")

    folders = os.listdir(train_folder)

    count = 0

    for folder in folders:
        bvh_path = os.path.join(train_folder, folder, f"{folder}.bvh")

        if os.path.exists(bvh_path):
            explore_bvh(bvh_path)
            count += 1

        if count >= 3:  # limit output
            break


# -------------------------------
# MAIN EXECUTION (IMPORTANT)
# -------------------------------
if __name__ == "__main__":

    # 🔥 OPTION 1: Test single BVH file
    single_file = "C:/Users/HP/Downloads/Motion-S-Text-to-Sign-Motion-Generation-Signvrse-main/motion-s-hierarchical-text-to-motion-generation-for-sign-language/Train/1433/1433.bvh"

    # 🔥 OPTION 2: Full dataset folder
    train_folder = "C:/Users/HP/Downloads/Motion-S-Text-to-Sign-Motion-Generation-Signvrse-main/motion-s-hierarchical-text-to-motion-generation-for-sign-language/Train"

    try:
        print("\n===== SINGLE FILE ANALYSIS =====")
        explore_bvh(single_file)
        print_multiple_frames(single_file, num_frames=3)

        print("\n\n===== MULTIPLE FILE ANALYSIS =====")
        explore_multiple_bvh(train_folder)

    except FileNotFoundError:
        print("❌ BVH file not found. Please check your path.")