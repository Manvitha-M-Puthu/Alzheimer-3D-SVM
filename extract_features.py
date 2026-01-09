import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from nilearn import datasets, image
from radiomics import featureextractor

# --- CONFIGURATION ---
DATA_DIR = "NIfTI_Data"
OUTPUT_CSV = "dementia_features.csv"
CLASSES = ["CN", "MCI", "AD"]

def create_reference_files():
    """Downloads template and creates a mask that EXACTLY matches the template size."""
    print("⬇️ Fetching Standard Brain Template & Atlas...")
    
    # 1. Get MNI152 Template (The 'Standard' Brain - usually 1mm resolution)
    target = datasets.load_mni152_template()
    target.to_filename("mni_template.nii.gz")
    
    # 2. Get Atlas for Hippocampus Mask (usually 2mm resolution)
    dataset_ho = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    
    # Handle whether nilearn returns a path or an object
    if isinstance(dataset_ho.maps, str):
        atlas_img = nib.load(dataset_ho.maps)
    else:
        atlas_img = dataset_ho.maps

    # --- FIX: RESAMPLE ATLAS TO MATCH TEMPLATE ---
    # The atlas (2mm) is smaller than the template (1mm). We must resize it.
    print("  ⚙️ Resampling Atlas to match Template resolution...")
    resampled_atlas = image.resample_to_img(
        source_img=atlas_img,
        target_img=target,
        interpolation='nearest' # Nearest neighbor to keep labels as integers (0, 1, 2...)
    )
    
    atlas_data = resampled_atlas.get_fdata()
    
    # 3. Create Binary Mask (Indices 9 & 20 are Hippocampus)
    mask_data = np.zeros(atlas_data.shape)
    mask_data[atlas_data == 9] = 1   # Left Hippo
    mask_data[atlas_data == 20] = 1  # Right Hippo
    
    # Save the new high-res mask
    mask_img = nib.Nifti1Image(mask_data, resampled_atlas.affine)
    nib.save(mask_img, "hippo_mask.nii.gz")
    
    print("  ✅ Template and Mask are now aligned.")
    return "mni_template.nii.gz", "hippo_mask.nii.gz"

def register_image(moving_path, fixed_path, output_path):
    """Aligns patient MRI (moving) to MNI Template (fixed)."""
    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)
    
    # Quick affine registration (Rigid + Scaling)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=50)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    final_transform = registration_method.Execute(fixed, moving)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(final_transform)
    aligned = resampler.Execute(moving)
    
    sitk.WriteImage(aligned, output_path)

def main():
    # Create the reference files first
    template_path, mask_path = create_reference_files()
    
    # Setup Radiomics Extractor
    # Note: 'resampledPixelSpacing': None ensures we don't accidentally resize again
    params = {'binWidth': 25, 'resampledPixelSpacing': None} 
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('shape')

    data_list = []
    
    print("\n🚀 Starting Feature Extraction Pipeline...")
    
    for group in CLASSES:
        search_path = os.path.join(DATA_DIR, group, "*.nii.gz")
        subjects = glob.glob(search_path)
        print(f"\nProcessing {group} group ({len(subjects)} scans found)...")
        
        for subject_path in subjects:
            temp_aligned = "temp_aligned.nii.gz"
            try:
                # 1. Register Subject to Template
                register_image(subject_path, template_path, temp_aligned)
                
                # 2. Extract Features
                # Now temp_aligned and mask_path should have identical dimensions
                feature_vector = extractor.execute(temp_aligned, mask_path)
                
                # 3. Store Data
                row = {'Subject': os.path.basename(subject_path), 'Label': group}
                for key, val in feature_vector.items():
                    if "original_" in key:
                        feature_name = key.replace("original_", "")
                        row[feature_name] = val
                
                data_list.append(row)
                print(f"  ✅ Processed: {row['Subject']}")
                
            except Exception as e:
                print(f"  ❌ Error processing {subject_path}: {e}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_aligned):
                    os.remove(temp_aligned)

    # Save to CSV
    if data_list:
        df = pd.DataFrame(data_list)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n🎉 Done! Features saved to {OUTPUT_CSV}")
    else:
        print("\n⚠️ No data was processed. Check your images.")

if __name__ == "__main__":
    main()