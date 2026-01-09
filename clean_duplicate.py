import os

# --- CONFIGURATION ---
DATA_DIR = "NIfTI_Data"  # Your main data folder
CLASSES = ["CN", "MCI", "AD"]

def remove_duplicates():
    print("🧹 Starting Duplicate Cleanup...")
    
    total_removed = 0
    
    for group in CLASSES:
        folder_path = os.path.join(DATA_DIR, group)
        if not os.path.exists(folder_path):
            continue
            
        print(f"\n📂 Scanning {group}...")
        
        # Get all files
        files = sorted(os.listdir(folder_path))
        
        # Dictionary to track unique patients
        seen_patients = {}
        
        for filename in files:
            if not filename.endswith(".nii") and not filename.endswith(".nii.gz"):
                continue
                
            # Extract Patient ID (e.g., "002_S_0295" from the long filename)
            # ADNI format usually starts with the ID
            try:
                # We assume the first 10 characters are the ID (e.g., 002_S_XXXX)
                # Adjust split logic if your files are named differently
                patient_id = filename.split("_ADNI")[0] 
                
                # Fallback if split fails
                if len(patient_id) < 3: 
                    patient_id = filename[:10] 
            except:
                patient_id = filename[:10]

            file_path = os.path.join(folder_path, filename)
            
            if patient_id in seen_patients:
                # We already have a scan for this patient! Delete this one.
                print(f"  ❌ Duplicate found for {patient_id}: Removing {filename}")
                os.remove(file_path)
                total_removed += 1
            else:
                # This is the first time seeing this patient. Keep it.
                seen_patients[patient_id] = file_path

    print(f"\n🎉 Cleanup Complete! Removed {total_removed} duplicate files.")

if __name__ == "__main__":
    remove_duplicates()