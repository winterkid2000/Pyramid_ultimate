import traceback
from totalsegmentator.python_api import totalsegmentator

def run_TS(dicom_dir, output_dir, organ):
    try:
        totalsegmentator(
            input=dicom_dir,
            output=output_dir,
            task="total",
            roi_subset=[organ]
        )
        return True
    except Exception as e:
        print(f"[Segmentator 오류] {e}")
        traceback.print_exc()
        return False
