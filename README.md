# 📷 Camera Calibration & Pose Estimation (PyTorch)

A from-scratch implementation of **camera calibration, intrinsic/extrinsic estimation, and 3D rotation modeling** using **PyTorch**, based on classical computer vision geometry and linear algebra.

This project demonstrates hands-on understanding of **camera projection models**, **Direct Linear Transformation (DLT)**, and **rotation matrix construction**, implemented under strict library constraints.

---

## 🚀 What This Project Does

✔️ Builds rotation matrices using Euler angles
✔️ Detects and refines checkerboard corners in image space
✔️ Constructs corresponding 3D world coordinates
✔️ Estimates **camera intrinsics** (fx, fy, cx, cy)
✔️ Recovers **camera extrinsics** (rotation + translation)
✔️ Implements the full **projection pipeline**:
[
\mathbf{x} = \mathbf{K}[\mathbf{R}|\mathbf{T}]\mathbf{X}
]

---

## 🧠 Core Concepts Demonstrated

* Camera geometry & pinhole camera model
* Euler-angle based rotations
* Orthogonality of rotation matrices
* Direct Linear Transformation (DLT)
* Singular Value Decomposition (SVD)
* Intrinsic / extrinsic matrix decomposition

---

## 🛠️ Tech Stack

* **Python**
* **PyTorch** (all math & linear algebra)
* **OpenCV** (corner detection only)
* **NumPy** (image interoperability)

> ⚠️ Implementation respects strict import constraints — no helper libraries or shortcuts.

All dependencies are listed in requirements.txt for reproducibility
---

## 📌 Implementation Overview

### 🔄 Rotation Estimation

* Computes forward (`xyz → XYZ`) and inverse (`XYZ → xyz`) rotations
* Uses Euler angles (degrees)
* Validates angle ranges to avoid gimbal lock
* Inverse rotation computed via matrix transpose

### 🎯 Corner Detection

* Detects a **7×3 checkerboard pattern**
* Applies sub-pixel refinement
* Selects **18 meaningful corners** across two perpendicular planes (XZ & YZ)

### 🌍 World Coordinate Mapping

* Constructs 3D points assuming:

  * Grid size: **10 mm**
  * Origin at plane intersection
* Output format: `(x, y, z)` in millimeters

### 📐 Intrinsic Calibration

* Builds projection matrix via **DLT**
* Extracts:

  * Focal lengths: `fx`, `fy`
  * Principal point: `(cx, cy)`

### 🧭 Extrinsic Calibration

* Decomposes projection matrix into:

  * Rotation matrix `R`
  * Translation vector `T`
* Ensures correct normalization and sign consistency

---

## 📂 Key Functions

| Function                      | Purpose                      |
| ----------------------------- | ---------------------------- |
| `findRot_xyz2XYZ`             | Euler-angle rotation matrix  |
| `findRot_XYZ2xyz`             | Inverse rotation             |
| `find_corner_img_coord`       | Image-space corner detection |
| `find_corner_world_coord`     | 3D world coordinates         |
| `find_intrinsic`              | Camera intrinsic estimation  |
| `find_extrinsic`              | Camera pose estimation       |
| `determine_projection_matrix` | DLT implementation           |
| `determine_K_matrix`          | Intrinsic matrix extraction  |

---

## 🧪 Input & Output

**Input**

* Image tensor: `3 × H × W` (torch tensor)
* Checkerboard with known geometry

**Output**

* Camera intrinsics: `fx, fy, cx, cy`
* Camera pose: rotation matrix `R`, translation vector `T`
* All outputs in **metric units**

---

## 💡 Why This Matters

This project shows:

* Ability to implement **core CV algorithms from scratch**
* Strong grasp of **linear algebra & geometry**
* Comfort working under **real-world constraints**
* Readiness for **computer vision, robotics, or AR/VR** roles

---

## 📈 Possible Extensions

* Radial & tangential distortion modeling
* Multi-image calibration (Zhang’s method)
* Reprojection error visualization
* Bundle adjustment optimization

---

## 🧑‍💻 Author

**Sudhanshu Iyer**
MS CS (AI/ML) | Computer Vision & Deep Learning
Former Product Engineer | Java · PyTorch · CV · Systems
