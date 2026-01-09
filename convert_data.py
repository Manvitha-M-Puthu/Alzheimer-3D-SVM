import os
import subprocess
import shutil

# --- CONFIGURATION ---
# Paths (relative to where this script is run)
RAW_DATA_DIR = "Raw_Data"
OUTPUT_DIR = "NIfTI_Data"
CONVERTER_TOOL = "./dcm2niix"  # On Windows, this points to dcm2niix.exe

# Classes to process
CLASSES = ["AD", "MCI", "CN"]

def convert_dcm_to_nii():
    # Check if converter exists
    if not os.path.exists(CONVERTER_TOOL) and not os.path.exists(CONVERTER_TOOL + ".exe"):
        print("❌ Error: dcm2niix tool not found! Please download it and place it in this folder.")
        return

    for group in CLASSES:
        print(f"\n--- Processing Group: {group} ---")
        
        # Define paths for this group
        group_raw_path = os.path.join(RAW_DATA_DIR, group)
        group_out_path = os.path.join(OUTPUT_DIR, group)
        
        # Create output directory if it doesn't exist
        os.makedirs(group_out_path, exist_ok=True)
        
        # Walk through all folders to find DICOMs
        # ADNI structure is Project -> Subject -> Scan Type -> Date -> ID -> DCM files
        for root, dirs, files in os.walk(group_raw_path):
            if not files:
                continue
            
            # Check if directory contains DICOM files (usually .dcm or no extension)
            # We assume a folder with > 10 files is a scan directory
            if len(files) > 10: 
                print(f"Converting scan in: {root}")
                
                # Run dcm2niix
                # -z y : Compress to .nii.gz
                # -f : Filename format (SubjectID_Protocol_Date)
                # -o : Output directory
                cmd = [
                    CONVERTER_TOOL,
                    "-z", "y",
                    "-f", "%i_%p_%t", 
                    "-o", group_out_path,
                    root
                ]
                
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL) # Hide generic output
                    print(f"  ✅ Converted.")
                except subprocess.CalledProcessError as e:
                    print(f"  ❌ Failed to convert {root}")

    print("\n🎉 All conversion complete! Check the 'NIfTI_Data' folder.")

if __name__ == "__main__":
    convert_dcm_to_nii()