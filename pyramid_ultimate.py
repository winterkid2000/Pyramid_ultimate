import multiprocessing
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath("."))

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

from segmentation.total import run_TS
from converter.meshconverter_nii import nii_mask_2_stl
from converter.bodyconverter import convert_dicom_to_nifti
from radiomics.shape import run_combined_descriptor
from model.external_patient import predict_with_model

class PyramidApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pyramid (PyRadioMics-Detector)")

        self._build_ui()

    def _build_ui(self):
        tk.Label(self.root, text="DICOM 폴더").grid(row=0, column=0, sticky="w")
        self.dicom_dir_entry = tk.Entry(self.root, width=60)
        self.dicom_dir_entry.grid(row=0, column=1)
        tk.Button(self.root, text="찾기", command=self.choose_dicom_dir).grid(row=0, column=2)

        tk.Label(self.root, text="출력 폴더").grid(row=1, column=0, sticky="w")
        self.output_dir_entry = tk.Entry(self.root, width=60)
        self.output_dir_entry.grid(row=1, column=1)
        tk.Button(self.root, text="찾기", command=self.choose_output_dir).grid(row=1, column=2)

        tk.Label(self.root, text="장기 이름").grid(row=2, column=0, sticky="w")
        organ_list = ['pancreas', 'lung', 'kidney', 'liver', 'spleen']
        self.organ_combobox = ttk.Combobox(self.root, values = organ_list, width=60)
        self.organ_combobox.grid(row = 2, column = 1, columnspan =2)
        self.organ_combobox.set('장기 선택')

        self.start_button = tk.Button(self.root, text="시작", command=self.start_pipeline)
        self.start_button.grid(row=3, column=0, columnspan=3, pady=10)

        self.log_output = scrolledtext.ScrolledText(self.root, height=15, width=80, state='disabled')
        self.log_output.grid(row=4, column=0, columnspan=3, padx=5, pady=5)

    def log(self, message):
        self.log_output.config(state='normal')
        self.log_output.insert(tk.END, message + "\n")
        self.log_output.config(state='disabled')
        self.log_output.see(tk.END)

    def choose_dicom_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.dicom_dir_entry.delete(0, tk.END)
            self.dicom_dir_entry.insert(0, os.path.normpath(path))

    def choose_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, os.path.normpath(path))

    def find_mask_file(self, output_dir, organ_name):
        files = os.listdir(output_dir)
        for f in files:
            if f.lower().endswith('.nii.gz') and organ_name.lower() in f.lower():
                return os.path.join(output_dir, f)
        return None

    def start_pipeline(self):
        dicom_path = self.dicom_dir_entry.get()
        out_path = self.output_dir_entry.get()
        organ = self.organ_combobox.get().strip().lower()

        if not all([dicom_path, out_path, organ]):
            messagebox.showwarning("하나라도 빼먹으면 서운해...")
            return

        os.makedirs(out_path, exist_ok=True)

        nifti_output = convert_dicom_to_nifti(dicom_path, out_path)
        if not nifti_output:
            return

        if not run_TS(dicom_path, out_path, organ):
            return

        nii_path = self.find_mask_file(out_path, organ)
        if not nii_path:
            self.log(f"{organ}에 해당하는 마스크 파일을 찾을 수 없다고...")
            return

        stl_path = os.path.join(out_path, f"{organ}.stl")

        if not nii_mask_2_stl(nii_path, stl_path):
            return

        if not isinstance(stl_path, (str, os.PathLike)) or not os.path.exists(stl_path):
            self.log(f'STL 파일 경로 오류...:{repr(stl_path)}')
            return
        
        output_xlsx = os.path.join(out_path, f"{organ}_merged_features.xlsx")
        if not run_combined_descriptor(nifti_output, nii_path, stl_path, output_xlsx):
            return

        model_path = resource_path('best_model_fold.pt')
        scaler_path = resource_path('scaler_fold.pkl')
        try:
            predict_with_model(output_xlsx, model_path, scaler_path, log_callback=self.log)
        except Exception as e:
            self.log(f"돌팔이었네...: {e}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == "--prevent-loop":
        sys.exit(0)
    root = tk.Tk()
    app = PyramidApp(root)
    root.mainloop()
