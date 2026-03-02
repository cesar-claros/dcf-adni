"""
Standalone entry point for ADNI data preprocessing.

Usage:
    python data_preprocessing.py
"""

import logging

from src.data_preprocessing import ADNIPreprocess

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    preprocessor = ADNIPreprocess(
        data_path="data/All_Subjects_My_Table_03Jul2025.csv",
        mri_path="data/UCSFFSX7_20Jun2025.csv",
        output_dir="data/",
    )
    preprocessor.run()
