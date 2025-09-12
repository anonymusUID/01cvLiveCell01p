import os
import numpy as np
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)

class SegmentationMetricsCalculator:
    def __init__(self):
        # Default paths
        self.default_mask_path = "/home/surajit/CV/Usupervised_Learning/Input/SKOV3_Phase_H4_mask/"
        self.default_segmented_path = "/home/surajit/CV/Usupervised_Learning/Output/SKOV3_Phase_H4/"
        self.default_output = "/home/surajit/CV/Usupervised_Learning/Output/SKOV3_Phase_H4/SKOV3_Phase_H4/"
        
        # Supported image extensions
        self.supported_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        
        # Metrics configuration
        self.metrics_config = {
            'dice': {'decimals': 4, 'name': 'Dice Coefficient'},
            'iou': {'decimals': 4, 'name': 'IoU'},
            'accuracy': {'decimals': 4, 'name': 'Accuracy'},
            'precision': {'decimals': 4, 'name': 'Precision'},
            'recall': {'decimals': 4, 'name': 'Recall'},
            'f1': {'decimals': 4, 'name': 'F1 Score'},
            'ssim': {'decimals': 4, 'name': 'SSIM'},
            'hausdorff': {'decimals': 2, 'name': 'Hausdorff Distance'}
        }

    def calculate_image_metrics(self, mask_path, seg_path):
        """Calculate metrics for a single image pair"""
        try:
            # Load images
            mask = imread(mask_path, as_gray=True)
            segmented = imread(seg_path, as_gray=True)
            
            # Verify dimensions
            if mask.shape != segmented.shape:
                print(f"Resizing segmented image to match mask dimensions: {mask.shape}")
                from skimage.transform import resize
                segmented = resize(segmented, mask.shape, anti_aliasing=True, preserve_range=True)
            
            # Binarize
            mask_bin = (mask > 0.5).astype(np.uint8)
            seg_bin = (segmented > 0.5).astype(np.uint8)
            
            # Calculate confusion matrix
            tp = np.sum((mask_bin == 1) & (seg_bin == 1))
            tn = np.sum((mask_bin == 0) & (seg_bin == 0))
            fp = np.sum((mask_bin == 0) & (seg_bin == 1))
            fn = np.sum((mask_bin == 1) & (seg_bin == 0))
            
            epsilon = 1e-7
            metrics = {
                'dice': (2 * tp) / (2 * tp + fp + fn + epsilon),
                'iou': tp / (tp + fp + fn + epsilon),
                'accuracy': (tp + tn) / (tp + tn + fp + fn + epsilon),
                'precision': tp / (tp + fp + epsilon),
                'recall': tp / (tp + fn + epsilon),
                'f1': (2 * tp) / (2 * tp + fp + fn + epsilon),
                'ssim': ssim(mask, segmented, data_range=segmented.max()-segmented.min()),
                'hausdorff': self.calculate_hausdorff(mask_bin, seg_bin)
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error processing {mask_path}: {str(e)}")
            return None

    def calculate_hausdorff(self, mask, segmented):
        """Safe Hausdorff distance calculation"""
        try:
            mask_coords = np.argwhere(mask)
            seg_coords = np.argwhere(segmented)
            
            if len(mask_coords) == 0 or len(seg_coords) == 0:
                return np.nan
                
            return max(directed_hausdorff(mask_coords, seg_coords)[0],
                     directed_hausdorff(seg_coords, mask_coords)[0])
        except:
            return np.nan

    def process_all_images(self, mask_path=None, segmented_path=None, output_prefix=None, max_workers=4):
        """Main processing function"""
        # Set paths
        mask_path = os.path.abspath(os.path.expanduser(mask_path or self.default_mask_path))
        segmented_path = os.path.abspath(os.path.expanduser(segmented_path or self.default_segmented_path))
        output_prefix = os.path.abspath(os.path.expanduser(output_prefix or self.default_output))
        
        # Verify directories
        if not os.path.isdir(mask_path):
            raise FileNotFoundError(f"Mask directory not found: {mask_path}")
        if not os.path.isdir(segmented_path):
            raise FileNotFoundError(f"Segmented directory not found: {segmented_path}")
        
        # Find all image files
        mask_files = [f for f in os.listdir(mask_path) 
                    if os.path.splitext(f)[1].lower() in self.supported_extensions]
        
        if not mask_files:
            raise FileNotFoundError(f"No supported image files found in {mask_path}")
        
        print(f"\nProcessing {len(mask_files)} mask files from: {mask_path}")
        print(f"Segmented images directory: {segmented_path}")
        
        # Process images
        results = []
        missing_segmented = []
        processing_errors = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for mask_file in mask_files:
                mask_full_path = os.path.join(mask_path, mask_file)
                base_name = os.path.splitext(mask_file)[0]
                
                # Try multiple possible segmented image locations
                possible_paths = [
                    os.path.join(segmented_path, base_name, mask_file),
                    os.path.join(segmented_path, mask_file),
                    os.path.join(segmented_path, f"{base_name}.png"),
                    os.path.join(segmented_path, f"{base_name}.jpg")
                ]
                
                seg_path = next((p for p in possible_paths if os.path.exists(p)), None)
                
                if not seg_path:
                    missing_segmented.append(mask_file)
                    continue
                
                futures.append(executor.submit(
                    self.calculate_image_metrics,
                    mask_full_path,
                    seg_path
                ))
            
            # Collect results
            for i, future in enumerate(tqdm(futures, desc="Processing images")):
                try:
                    result = future.result()
                    if result:
                        results.append({
                            'image': mask_files[i],
                            'mask_path': os.path.join(mask_path, mask_files[i]),
                            'segmented_path': next(p for p in possible_paths if os.path.exists(p)),
                            **result
                        })
                except Exception as e:
                    processing_errors.append(str(e))
        
        # Save results
        if results:
            self.save_results(results, output_prefix)
            self.print_summary(results, missing_segmented, processing_errors)
        else:
            print("No valid results to save")
        
        return results

    def save_results(self, results, output_prefix):
        """Save results to CSV files"""
        df = pd.DataFrame(results)
        
        # Round metrics
        for metric in self.metrics_config:
            if metric in df.columns:
                df[metric] = df[metric].round(self.metrics_config[metric]['decimals'])
        
        # Save detailed results
        detailed_csv = f"{output_prefix}_detailed.csv"
        df.to_csv(detailed_csv, index=False)
        print(f"\nDetailed metrics saved to: {detailed_csv}")
        
        # Calculate and save averages and standard deviations
        numeric_cols = [col for col in df.columns if col not in ['image', 'mask_path', 'segmented_path']]
        avg_metrics = df[numeric_cols].mean().to_frame().T
        std_metrics = df[numeric_cols].std().to_frame().T
        
        # Combine averages and standard deviations
        summary_df = pd.concat([avg_metrics, std_metrics], keys=['Average', 'Std Dev'])
        
        summary_csv = f"{output_prefix}_summary.csv"
        summary_df.to_csv(summary_csv)
        print(f"Summary metrics (average ± std dev) saved to: {summary_csv}")

    def print_summary(self, results, missing_segmented, processing_errors):
        """Print summary statistics"""
        df = pd.DataFrame(results)
        numeric_cols = [col for col in df.columns if col not in ['image', 'mask_path', 'segmented_path']]
        avg_metrics = df[numeric_cols].mean()
        std_metrics = df[numeric_cols].std()
        
        print("\nSUMMARY STATISTICS")
        print("==================")
        print(f"Total images processed: {len(results)}")
        print(f"Missing segmented images: {len(missing_segmented)}")
        print(f"Processing errors: {len(processing_errors)}")
        
        print("\nMETRICS (Average ± Standard Deviation):")
        print("--------------------------------------")
        for metric in self.metrics_config:
            if metric in avg_metrics:
                avg = avg_metrics[metric]
                std = std_metrics[metric]
                decimals = self.metrics_config[metric]['decimals']
                print(f"{self.metrics_config[metric]['name']:<20}: {avg:.{decimals}f} ± {std:.{decimals}f}")
        
        if missing_segmented:
            print("\nFirst 5 missing segmented images:")
            for img in missing_segmented[:5]:
                print(f"  - {img}")
            if len(missing_segmented) > 5:
                print(f"  - ...and {len(missing_segmented)-5} more")

if __name__ == "__main__":
    calculator = SegmentationMetricsCalculator()
    
    try:
        results = calculator.process_all_images()
    except Exception as e:
        print(f"\nError: {str(e)}")