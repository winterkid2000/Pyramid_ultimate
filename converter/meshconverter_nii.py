import os
import numpy as np
import nibabel as nib
from skimage import measure
from stl import mesh

def nii_mask_2_stl(nifti_path: str, stl_path: str, threshold: float = 0) -> bool:
    try:
        if not os.path.exists(nifti_path):
            print(f"nii 마스크 찾아봐라: {nifti_path}")
            return False

        img = nib.load(nifti_path)
        data = img.get_fdata()
        affine = img.affine

        if np.max(data) < threshold:
            print(f"마스크 없네잉: {nifti_path}")
            return False

        verts, faces, _, _ = measure.marching_cubes(data, level=threshold)

        verts_hom = np.c_[verts, np.ones(verts.shape[0])]
        verts_world = (affine @ verts_hom.T).T[:, :3]

        stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = verts_world[face[j], :]

        os.makedirs(os.path.dirname(stl_path), exist_ok=True)
        stl_mesh.save(stl_path)
        return True

    except Exception as e:
        print(f"Mesh에서 문제 생겼으니까 빨리 디버깅 해!!!!!!!!!!!!!!!: {e}")
        return False
