# 📷 Camera Calibration & Pose Estimation (PyTorch)

A from-scratch implementation of **camera calibration and pose estimation** using **PyTorch**, based on classical computer vision geometry.

The project estimates **camera intrinsics and extrinsics** from a checkerboard pattern and implements **rotation matrices** using Euler angles, demonstrating a full pinhole camera pipeline without relying on high-level calibration APIs.

---

## ✨ Highlights

- Euler-angle rotation matrices (xyz ↔ XYZ)
- Checkerboard corner detection with sub-pixel refinement
- 3D world coordinate construction
- Projection matrix estimation using **DLT + SVD**
- Camera intrinsic parameter recovery (fx, fy, cx, cy)
- Camera pose estimation (R, T)

---

## 🧠 Core Concepts Demonstrated

- Camera projection model
- Rotation matrices & orthogonality
- Direct Linear Transformation (DLT)
- Singular Value Decomposition (SVD)
- Intrinsic / extrinsic matrix decomposition

---

## 🛠️ Tech Stack

- **Python**
- **PyTorch**
- **OpenCV** (corner detection only)
- **NumPy**

> All core math and geometry are implemented using **PyTorch tensors**.

---

All dependencies are listed in `requirements.txt` for reproducibility.

---

## Example Usage

The calibration logic is implemented in `geometry.py` and can be executed
using the provided evaluation scripts.

### Camera Calibration
```bash
python evaluate_calibration.py --input images/chessboard.png
