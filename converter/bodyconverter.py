import os
import dicom2nifti

def convert_dicom_to_nifti(dicom_dir, output_dir):
    try:
        dicom2nifti.convert_directory(dicom_dir, output_dir, reorient=True)

        nifti_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz')]
        if not nifti_files:
            print("출력 폴더에 .nii.gz 파일이 없습니다.")
            return None

        nifti_files.sort(key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), reverse=True)
        output_path = os.path.join(output_dir, nifti_files[0])

        print(f"변환 성공: {output_path}")
        return output_path 

    except Exception as e:
        print(f"변환 중 오류 발생: {e}")
        return None
