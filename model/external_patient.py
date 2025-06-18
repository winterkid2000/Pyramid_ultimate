import pandas as pd
import torch
import pickle
from model.FFT_model import FTTransformer
import pydicom
import os 
FEATURE_COLS = [
    'Integral_Total_HU', 'Kurtosis', 'Max_HU', 'Mean_HU', 'Median_HU', 'Min_HU',
    'Skewness', 'Sphere_Diameter', 'HU_STD', 'Total_HU', 'Volume', "Surface Area", 
    'Convex_Hull_Ratio', 'Sphericity', 'Major_Axis', 'Minor_Axis',
    'Eccentricity', 'Fourier_Very_Low', 'Fourier_Low', 'Fourier_Mid_Low',
    'Fourier_Mid_High', 'Fourier_High', 'HU_Histogram_1', 'HU_Histogram_2',
    'HU_Histogram_3', 'HU_Histogram_4', 'HU_Histogram_5'
]

def get_info_patient(dicom_path):
    dicoms = [f for f in os.listdir(dicom_path) if f.endswith('.dcm')]
    acquired_dicom = dicoms[0]
    file = os.path.join(dicom_path, acquired_dicom)

    info = pydicom.dcmread(file)
    Patient_name = str(info.PatientName)
    Patient_number = info.PatientID

    return Patient_name, Patient_number

def predict_with_model(xlsx_path, model_path, scaler_path, dicom_path, log_callback=None):
    try:

        df = pd.read_excel(xlsx_path)

        model = FTTransformer(input_dim=len(FEATURE_COLS))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        X = df[FEATURE_COLS].values
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            probs = model(X_tensor).numpy().squeeze()
        patient_name, patient_number = get_info_patient(dicom_path)    

        if probs.ndim == 0:
            df["Prediction"] = ["비정상" if probs > 0.5 else "정상"]
            df["Probability"] = [float(probs)]
            probs1 = float(probs)*100
        else:
            df["Prediction"] = ["비정상" if p > 0.5 else "정상" for p in probs]
            df["Probability"] = probs
            probs1 = float(probs[0])*100
        if log_callback:
            result = "비정상" if probs1 > 50 else "정상"
            log_callback("환자 번호 | 환자 이름 | 결과 | 비정상도(%) | 판단 기준")
            log_callback("----------------------------------------------")
            log_callback(f"{patient_number}   {patient_name} {result} {probs1:.2f} %   50% <")

        df.to_excel(xlsx_path, index=False)

    except Exception as e:
        if log_callback:
            log_callback(f"돌팔이었습니다...: {str(e)}")
        return False
