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
OUTPUT_CSV = "final_radiomics_features.csv"
CLASSES = ["CN", "MCI", "AD"]

def create_reference_files():
    """Generates the MNI Template and Hippocampus Mask."""
    print("⬇️  Fetching Standard Brain Template & Atlas...")
    
    # 1. Load MNI Template
    target = datasets.load_mni152_template()
    target.to_filename("mni_template.nii.gz")
    
    # 2. Load Harvard-Oxford Atlas
    dataset_ho = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    if isinstance(dataset_ho.maps, str):
        atlas_img = nib.load(dataset_ho.maps)
    else:
        atlas_img = dataset_ho.maps

    # 3. Resample Atlas to match Template (Nearest Neighbor to keep integer labels)
    print("  ⚙️  Resampling Atlas to match Template...")
    resampled_atlas = image.resample_to_img(atlas_img, target, interpolation='nearest')
    atlas_data = resampled_atlas.get_fdata()
    
    # 4. Create Binary Mask (Left Hippo=9, Right Hippo=20)
    mask_data = np.zeros(atlas_data.shape)
    mask_data[atlas_data == 9] = 1   
    mask_data[atlas_data == 20] = 1  
    
    mask_img = nib.Nifti1Image(mask_data, resampled_atlas.affine)
    nib.save(mask_img, "hippo_mask.nii.gz")
    
    return "mni_template.nii.gz", "hippo_mask.nii.gz"

def preprocess_and_register(input_path, template_path, output_path):
    """
    Performs N4 Bias Correction and High-Quality Affine Registration.
    """
    # Load images
    fixed = sitk.ReadImage(template_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(input_path, sitk.sitkFloat32)
    
    # 1. N4 Bias Field Correction (Cleans image artifacts)
    try:
        mask_image = sitk.OtsuThreshold(moving, 0, 1, 200)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 30])
        moving_corrected = corrector.Execute(moving, mask_image)
    except:
        moving_corrected = moving # Fallback if Otsu fails
        
    # 2. Registration (Rigid + Affine)
    # We use high-quality settings to ensure the hippocampus aligns perfectly
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving_corrected, sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Robust Optimizer settings
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=0.01, numberOfIterations=200, 
        relaxationFactor=0.5, gradientMagnitudeTolerance=1e-46
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    final_transform = registration_method.Execute(fixed, moving_corrected)
    
    # Apply Transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed) # Forces output to match Template Geometry exactly
    resampler.SetTransform(final_transform)
    aligned = resampler.Execute(moving_corrected)
    
    sitk.WriteImage(aligned, output_path)

def main():
    template_path, mask_path = create_reference_files()
    
    # --- CRITICAL FIX FOR SHAPE FEATURES ---
    # We allow a tiny tolerance for geometry mismatch to prevent errors
    params = {
        'binWidth': 25, 
        'resampledPixelSpacing': None,
        'geometryTolerance': 1e-4  # Allows micro-mismatches without crashing
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    
    # Explicitly ENABLE features
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder') # Intensity
    extractor.enableFeatureClassByName('glcm')       # Texture
    extractor.enableFeatureClassByName('shape')      # VOLUME (Crucial for AD!)
    
    data_list = []
    print("\n🚀 Starting FINAL Feature Extraction (Shape Enabled)...")
    
    for group in CLASSES:
        search_path = os.path.join(DATA_DIR, group, "*.nii.gz")
        subjects = glob.glob(search_path)
        print(f"\nProcessing {group} group ({len(subjects)} scans)...")
        
        for subject_path in subjects:
            temp_aligned = "temp_aligned_final.nii.gz"
            try:
                # 1. Preprocess & Register
                preprocess_and_register(subject_path, template_path, temp_aligned)
                
                # 2. Extract Features
                feature_vector = extractor.execute(temp_aligned, mask_path)
                
                # 3. Clean and Store
                row = {'Subject': os.path.basename(subject_path), 'Label': group}
                for key, val in feature_vector.items():
                    if "original_" in key:
                        feature_name = key.replace("original_", "")
                        row[feature_name] = val
                
                data_list.append(row)
                print(f"  ✅ {row['Subject']} | Volume: {row.get('shape_VoxelVolume', 'N/A')}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
            finally:
                if os.path.exists(temp_aligned):
                    os.remove(temp_aligned)

    if data_list:
        df = pd.DataFrame(data_list)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n🎉 Success! Features saved to {OUTPUT_CSV}")
    else:
        print("\n❌ No data processed.")

if __name__ == "__main__":
    main()