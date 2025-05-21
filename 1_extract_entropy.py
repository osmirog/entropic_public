import os
import glob
import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor
import sys
import pandas as pd
import time
import yaml
import re

# Global configuration for file paths
PARAM_PATH = '/home/osmirog/Projects/thesis/entropic/pyradiomics/examples/exampleSettings/Params.yaml'

# Load parameters from the YAML file
with open(PARAM_PATH, 'r') as param_file:
    params = yaml.safe_load(param_file)

"""
Checks if the specified data folder and necessary subdirectories exist.
Prints warnings for any missing directories.
"""
def validate_directories(data_folder):
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"The specified data folder '{data_folder}' does not exist.")

    for patient_dir in glob.glob(os.path.join(data_folder, '*')):
        if os.path.isdir(patient_dir):
            dicom_dir = os.path.join(patient_dir, '000000', '00000001')
            roi_dir = os.path.join(patient_dir, 'RoiVolume')

            if not os.path.exists(dicom_dir):
                print(f"The DICOM directory '{dicom_dir}' does not exist for patient '{patient_dir}'.")

            if not os.path.exists(roi_dir):
                print(f"The ROI directory '{roi_dir}' does not exist for patient '{patient_dir}'.")

            if not glob.glob(os.path.join(roi_dir, '*.nii.gz')):
                print(f"No ROI files found in '{roi_dir}' for patient '{patient_dir}'.")

"""
Processes each patient's data: reads the DICOM series and the ROI file, extracts the entropy
feature using PyRadiomics, and saves the results.
"""
def process_patient(patient_dir, extractor, output_dir, summary_data):
    # Inner function to extract and save entropy features
    def extract_features(ct_scan, roi_file, extractor, lesion_label):
        correction = 0
        try:
            features = extractor.execute(ct_scan, roi_file, voxelBased=True)
        except ValueError:  # Correct Image/Mask geometry mismatch
            
            try:
                extractor.settings['correctMask'] = True
                features = extractor.execute(ct_scan, roi_file, voxelBased=True)
                extractor.settings['correctMask'] = False
                correction = 1
            except ValueError:
                print(f"{lastname}: Bounding box of ROI is larger than image space. Skipping the ROI.")
                return 

        entropy_key = 'original_firstorder_Entropy'
        if entropy_key in features:
            entropy_array = sitk.GetArrayFromImage(features[entropy_key])
            # Use patient folder name as identifier and a fixed lesion label (here: "ROI")
            npy_filename = f"{lastname}_{lesion_label}_entropy.npy"
            np.save(os.path.join(output_dir, npy_filename), entropy_array)

            entropy_nrrd = features[entropy_key]
            nrrd_filename = f"{lastname}_{lesion_label}_entropy.nrrd"
            sitk.WriteImage(entropy_nrrd, os.path.join(output_dir, nrrd_filename), True)

            entropy_value = np.mean(entropy_array)
            summary_data.append([npy_filename[:-4], entropy_value, correction])
            print(f"Lesion {lesion_label}: Entropy = {entropy_value}")

    start_time = time.time()  # Start timing
    lastname = os.path.basename(patient_dir)
    print(f"Processing {lastname}")
    dicom_dir = os.path.join(patient_dir, '000000', '00000001')
    roi_dir = os.path.join(patient_dir, 'RoiVolume')
    
    # Get the (single) ROI file from the RoiVolume folder
    roi_files = glob.glob(os.path.join(roi_dir, '*.nii.gz'))
    if not roi_files:
        print(f"No ROI file found in {roi_dir} for patient {lastname}.")
        return
    roi_file = roi_files[0]

    # Read DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_series)
    ct_scan = reader.Execute()

    # Extract features for the only lesion/ROI provided
    lesion_label = "ROI"
    extract_features(ct_scan, roi_file, extractor, lesion_label)

    end_time = time.time()  # End timing
    elapsed_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"{lastname} processing time: {elapsed_time_ms:.2f} ms")


"""
Main function that initializes the feature extractor, loops over a range of kernel_radius
values, processes each patient, and writes the summary data to CSV files.
"""
def main(data_folder, base_output_folder):
    start_total_time = time.time()
    # Check that data folders exist
    validate_directories(data_folder)

    # Define a range of kernel_radius values to explore.
    # Chosen values are small to avoid oversmoothing finer details.
    kernel_radius_values = [1,2,3,4]
    
    # Loop through each kernel_radius value
    for kernel_radius in kernel_radius_values:
        print(f"\nProcessing with kernel_radius = {kernel_radius}")

        # Initialize radiomics feature extractor for the current kernel_radius
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        # Pass the kernelRadius as a list
        extractor.enableFeaturesByName(firstorder=['Entropy'], kernelRadius=[kernel_radius])
        
        # Manual settings  
        extractor.settings['resampledPixelSpacing'] = [1, 1, 1]
        extractor.settings['binWidth'] = 5 
        
        # Optionally, load parameters from the YAML file:
        # extractor.loadParams(PARAM_PATH)
        print("Current extractor settings:")
        for key, value in extractor.settings.items():
            print(f"{key}: {value}")

        # Create a dedicated output folder for this kernel_radius run
        output_folder = f"{base_output_folder}_kernelRadius_{kernel_radius}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        summary_data = []

        # Process each patient's data
        for patient_dir in glob.glob(os.path.join(data_folder, '*')):
                if os.path.isdir(patient_dir): 
                    dicom_dir = os.path.join(patient_dir, '000000', '00000001')
                    roi_dir = os.path.join(patient_dir, 'RoiVolume')

                if not os.path.exists(dicom_dir):
                    print(f"The DICOM directory '{dicom_dir}' does not exist for patient '{patient_dir}'.")
                    continue
                if not os.path.exists(roi_dir):
                    print(f"The ROI directory '{roi_dir}' does not exist for patient '{patient_dir}'.")
                    continue
                if not glob.glob(os.path.join(roi_dir, '*.nii.gz')):
                    print(f"No ROI files found in '{roi_dir}' for patient '{patient_dir}'.")
                    continue
                
                process_patient(patient_dir, extractor, output_folder, summary_data)

        # Write summary data to CSV for this kernel_radius value
        summary_df = pd.DataFrame(summary_data, columns=['File Name', 'Entropy Value', 'Correction'])
        summary_csv = os.path.join(output_folder, 'entropy_average.csv')
        summary_df.to_csv(summary_csv, index=False)
        print(f"Saved summary for kernel_radius {kernel_radius} to {summary_csv}")

    end_total_time = time.time()  # End timing
    elapsed_time_sec = (end_total_time - start_total_time)
    print(f"\nProcessing finished. Total processing time: {elapsed_time_sec:.2f} seconds")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_data_folder> [path_to_output_folder]")
        sys.exit(1)

    data_folder = sys.argv[1]
    base_output_folder = sys.argv[2] if len(sys.argv) > 2 else 'extract_entropy_results'
    main(data_folder, base_output_folder)
