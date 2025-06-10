import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft
import trimesh

very_low = (0, 24)
low = (24, 48)
mid_low = (48, 72)
mid_high = (72, 96)
high = (96, 121)

def fode(fd):
    total = sum([
        np.mean(fd[very_low[0]:very_low[1]]),
        np.mean(fd[low[0]:low[1]]),
        np.mean(fd[mid_low[0]:mid_low[1]]),
        np.mean(fd[mid_high[0]:mid_high[1]]),
        np.mean(fd[high[0]:high[1]]),
    ])
    return [np.mean(fd[a:b]) / total if total else 0 for a, b in [very_low, low, mid_low, mid_high, high]]

def run_combined_descriptor(nifti_ct_path, nifti_mask_path, stl_path, output_csv_path) -> bool:
    try:
        if not isinstance(stl_path, (str, os.PathLike)):
            return False

        ct_img = nib.load(nifti_ct_path)
        mask_img = nib.load(nifti_mask_path)

        ct = ct_img.get_fdata()
        mask = mask_img.get_fdata()

        if ct.shape != mask.shape:
            return False

        hu_values = ct[mask > 0]
        hu_values = hu_values[np.isfinite(hu_values)].astype(np.int16)

        if len(hu_values) == 0:
            return False

        voxel_volume = np.prod(ct_img.header.get_zooms())
        voxel_count = np.sum(mask > 0)
        volume = voxel_count * voxel_volume

        hu_features = {
            'Patient N': os.path.basename(stl_path),
            "Integral_Total_HU": np.sum(hu_values) * 0.001,
            "Kurtosis": kurtosis(hu_values),
            "Max_HU": np.max(hu_values),
            "Mean_HU": np.mean(hu_values),
            "Median_HU": np.median(hu_values),
            "Min_HU": np.min(hu_values),
            "Skewness": skew(hu_values),
            "Sphere_Diameter": 2 * ((3 * volume) / (4 * np.pi)) ** (1 / 3) * 0.1,
            "HU_STD": np.std(hu_values),
            "Total_HU": np.sum(hu_values) * voxel_volume
        }

        hu_hist_features = {}
        total_voxels = len(hu_values)
        bins = [(-151, -61), (-60, 30), (31, 120), (121, 210), (211, 300)]
        for i, (low, high) in enumerate(bins, start=1):
            count = np.sum((hu_values >= low) & (hu_values <= high))
            percentage = (count / total_voxels * 100) if total_voxels > 0 else 0
            hu_hist_features[f"HU_Histogram_{i}"] = percentage

        mesh = trimesh.load(stl_path, file_type='stl', force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        volume = mesh.volume
        surface = mesh.area
        convex = mesh.convex_hull.volume
        sphericity = (np.pi ** (1/3) * (6 * volume)**(2/3)) / surface if surface > 0 else 0

        inertia = mesh.moment_inertia
        eigval, _ = np.linalg.eigh(inertia)
        major, minor = max(eigval), min(eigval)
        ecc = np.sqrt(1 - minor / major) if major > 0 else 0

        verts = np.array([v[0] + 1j * v[1] for v in mesh.vertices])
        fd = np.abs(fft(verts))[:121]
        ratios = fode(fd)

        shape_features = {
            'Volume': volume,
            'Surface Area': surface,
            'Convex_Hull_Ratio': volume / convex if convex > 0 else 0,
            'Sphericity': sphericity,
            'Major_Axis': major,
            'Minor_Axis': minor,
            'Eccentricity': ecc,
            'Fourier_Very_Low': ratios[0],
            'Fourier_Low': ratios[1],
            'Fourier_Mid_Low': ratios[2],
            'Fourier_Mid_High': ratios[3],
            'Fourier_High': ratios[4],
        }

        merged = {**hu_features, **shape_features, **hu_hist_features}
        df = pd.DataFrame([merged])
        df.to_excel(output_csv_path, index=False)

        return True

    except Exception:
        return False
