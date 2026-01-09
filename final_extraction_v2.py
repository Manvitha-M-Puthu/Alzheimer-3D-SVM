import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from nilearn import datasets, image
from radiomics import featureextractor

DATA_DIR = "NIfTI_Data"
OUTPUT_CSV = "final_features_with_scaling.csv"
CLASSES = ["CN", "MCI", "AD"]

def create_reference_files():
    print("⬇️  Fetching Standard Brain Template & Atlas...")
    target = datasets.load_mni152_template()
    target.to_filename("mni_template.nii.gz")
    
    dataset_ho = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    if isinstance(dataset_ho.maps, str):
        atlas_img = nib.load(dataset_ho.maps)
    else:
        atlas_img = dataset_ho.maps

    print("  ⚙️  Resampling Atlas to match Template...")
    resampled_atlas = image.resample_to_img(atlas_img, target, interpolation='nearest')
    atlas_data = resampled_atlas.get_fdata()
    
    mask_data = np.zeros(atlas_data.shape)
    mask_data[atlas_data == 9] = 1   
    mask_data[atlas_data == 20] = 1  
    
    mask_img = nib.Nifti1Image(mask_data, resampled_atlas.affine)
    nib.save(mask_img, "hippo_mask.nii.gz")
    
    return "mni_template.nii.gz", "hippo_mask.nii.gz"

def two_stage_registration(input_path, template_path, output_path):
    
    fixed = sitk.ReadImage(template_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(input_path, sitk.sitkFloat32)
    
    
    init_rigid = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    reg_rigid = sitk.ImageRegistrationMethod()
    reg_rigid.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg_rigid.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=0.01, numberOfIterations=100)
    reg_rigid.SetInitialTransform(init_rigid, inPlace=False)
    reg_rigid.SetInterpolator(sitk.sitkLinear)
    
    final_rigid = reg_rigid.Execute(fixed, moving)
    
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(final_rigid)
    moving_straight = resampler.Execute(moving)
    
    
    reg_affine = sitk.ImageRegistrationMethod()
    reg_affine.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg_affine.SetOptimizerAsRegularStepGradientDescent(learningRate=0.1, minStep=0.001, numberOfIterations=100)
    reg_affine.SetInitialTransform(sitk.AffineTransform(3), inPlace=False) # Start from identity
    reg_affine.SetInterpolator(sitk.sitkLinear)
    
    final_affine = reg_affine.Execute(fixed, moving_straight)
    
    
    resampler.SetTransform(final_affine)
    final_image = resampler.Execute(moving_straight)
    sitk.WriteImage(final_image, output_path)
    
    return final_affine

def main():
    template_path, mask_path = create_reference_files()
    
    params = {'binWidth': 25, 'resampledPixelSpacing': None, 'geometryTolerance': 1e-3}
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    
    data_list = []
    print("\n🚀 Starting Feature Extraction (v4 - Two-Stage Robust)...")
    
    for group in CLASSES:
        search_path = os.path.join(DATA_DIR, group, "*.nii.gz")
        subjects = glob.glob(search_path)
        print(f"\nProcessing {group} group ({len(subjects)} scans)...")
        
        for subject_path in subjects:
            temp_aligned = "temp_aligned_v4.nii.gz"
            try:
                # 1. Run Two-Stage Registration
                transform = two_stage_registration(subject_path, template_path, temp_aligned)
                
                # 2. Calculate Scaling Factor (The "Atrophy" signal)
                matrix = transform.GetParameters()[:9] 
                matrix_np = np.array(matrix).reshape(3, 3)
                scaling_factor = np.linalg.det(matrix_np)
                
                # 3. Extract Features
                feature_vector = extractor.execute(temp_aligned, mask_path)
                
                # 4. Store Data
                row = {'Subject': os.path.basename(subject_path), 'Label': group}
                row['scaling_factor'] = scaling_factor
                
                for key, val in feature_vector.items():
                    if "original_" in key:
                        feature_name = key.replace("original_", "")
                        row[feature_name] = val
                
                data_list.append(row)
                print(f"  ✅ {row['Subject']} | Stretch Factor: {scaling_factor:.4f}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
            finally:
                if os.path.exists(temp_aligned):
                    os.remove(temp_aligned)

    if data_list:
        df = pd.DataFrame(data_list)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n🎉 Success! Features saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()