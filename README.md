
# Entropic Phenotyping of Colorectal Liver Metastases

**NOTE: THIS DOCUMENT IS AI-GENERATED BASED ON THE CODE REPOSITORY AND RESEARCH PAPER**

## Entropic: Voxel-Wise Tumor Entropy Mapping & Prognostic Clustering for CRLM

This repository contains the code and notebooks implementing a fully automated pipeline to:

1. **Extract** 3D, voxel‐wise local entropy maps from portal-phase CT of colorectal liver metastases (CRLM).  
2. **Cluster** patient‐specific entropy volumes using a spatially informed Image Euclidean Distance (IMED) metric and multiple unsupervised algorithms.  
3. **Analyze** the association between entropy‐derived clusters and clinical outcomes (OS, DFS) via univariate and multivariate survival models.

By embedding tumor heterogeneity into an integrated prognostic framework, this workflow lays the groundwork for CT-based “virtual biopsies” and personalized risk stratification in precision oncology.


## 📦 Repository Structure

```
.
├── 1_extract_entropy.py

├── 2_clustering.ipynb

├── 3_statistical_analysis.ipynb

├── example_settings/

│   └── Params.yaml

├── data/

│   ├── raw_CT/ ← DICOM series & ROI masks

│   └── DB_AIRCl.csv ← Patient‐ID ⇆ name mapping

└── README.md
```

---

## 🛠️ Requirements

- **Python 3.8+**  
- **PyRadiomics**, **SimpleITK**, **numpy**, **pandas**, **PyYAML**  
- **scikit-image**, **scikit-learn**, **scikit-learn-extra**, **lifelines**, **scipy**  
- **openpyxl**, **matplotlib**, **torch**

Install via:

```bash
pip install pyradiomics SimpleITK numpy pandas pyyaml scikit-image scikit-learn scikit-learn-extra lifelines scipy openpyxl matplotlib torch
```

---

## ⚙️ Phase 1: Entropy Extraction

**Script:** `1_extract_entropy.py`

### Purpose:

- Read each patient’s portal‐phase CT DICOM series and corresponding segmented ROI (NIfTI).
- Compute a voxel‐wise first‐order entropy map using PyRadiomics (kernel radius configurable).
- Save each entropy volume as both `.npy` (NumPy array) and `.nrrd` (ITK image), and record the global mean entropy in a summary CSV.

### Usage:

```bash
python 1_extract_entropy.py \
    /path/to/data_folder         \
    /path/to/output_folder
```

- `data_folder` should contain subdirectories per patient, each with:
  ```
  ├── 000000/00000001/   ← DICOM series
  └── RoiVolume/         ← *.nii.gz ROI masks
  ```
- `output_folder` will be created (if needed) as:
  ```
  output_folder_kernelRadius_<kr>/
  ```

### Key Parameters (in‐script):

- `kernel_radius_values` – list of PyRadiomics kernel radii to explore (e.g., `[1,2,3,4]`)
- `resampledPixelSpacing` – voxel spacing (e.g., `[1,1,1]`)
- `binWidth` – gray‐level discretization (e.g., `5`)

---

## ⚙️ Phase 2: Clustering of Entropy Maps

**Notebook:** `2_clustering.ipynb`

### Purpose:

- Map each entropy file to a patient ID via `DB_AIRCl.csv`.
- Downsample each 3D entropy volume for computational efficiency.
- Compute the Image Euclidean Distance (IMED) pairwise distance matrix for each `(kernel_radius, σ)` combination:
  - Builds separable Gaussian covariance kernels for each dimension.
  - Kronecker‐combines them into the full metric matrix.
  - Applies the square‐root factor (standardizing transform) to each vectorized volume.
- Apply four clustering algorithms (K-Means, K-Medoids, Agglomerative, DBSCAN) across a grid of:
  - `kernel_radius ∈ {1,2,3,4}`
  - `σ ∈ {1,2,3}`
  - `n_clusters ∈ {2,3,4,5}`
- Export for each combination:
  - `distance_matrix_kRad_{kr}_sigma_{σ}.npy`
  - `clustering_results_kRad_{kr}_sigma_{σ}_nClusters_{k}.{npy,csv}` (merged with clinical IDs)

### To Run:

- Open and execute all cells in order.
- Adjust the top‐level configuration block for your local paths and hyperparameter grid.

---

## ⚙️ Phase 3: Statistical Analysis

**Notebook:** `3_statistical_analysis.ipynb`

### Purpose:

- **Univariate tests:**
  - Log-rank tests on OS and DFS for each clustering stratification.
  - χ² / Fisher’s exact tests for categorical clinical covariates.
  - Kruskal–Wallis tests for numeric variables.
- **Multiple‐testing correction:**
  - Bonferroni adjustment across families of tests.
  - Highlight significant p-values and small‐cluster warnings in an Excel report.
- **Kaplan–Meier visualization:**
  - Plot survival curves for selected cluster configurations.
- **Cox proportional hazards modeling:**
  - Fit multivariate models combining cluster membership and key clinical covariates.
  - Report hazard ratios, confidence intervals, C-index, and proportionality tests.

### To Run:

- Execute all cells sequentially.
- Ensure that `clustering_results/` and the patient database are accessible.

---

## 📄 Reference Study

This pipeline underpins the manuscript:

> “Voxel-Wise Tumor Entropy Mapping and IMED-Clustering for Prognostic Stratification in Colorectal Liver Metastases”

Please cite our study when using or adapting this code.

---

## 🚀 Getting Started

1. Place your DICOM & ROI data under `data/raw_CT/`.
2. Update `PARAM_PATH` in `1_extract_entropy.py` (or point to your own YAML).
3. Run Phase 1 to generate entropy maps.
4. Configure file paths in `2_clustering.ipynb` and run Phase 2.
5. Open `3_statistical_analysis.ipynb`, adjust paths if needed, and run Phase 3.

Enjoy exploring tumor heterogeneity with **Entropic**!
```
