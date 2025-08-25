import numpy as np
import os
import cv2
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import traceback

# Define input and output directories
rgb_dir = '/data1/code/ossutil-v1.7.19-linux-amd64/rgb_unseen_test/train/imageso'
ir_dir = '/data1/code/ossutil-v1.7.19-linux-amd64/rgb_unseen_test/train/iro'
output_dir = '/data1/code/ossutil-v1.7.19-linux-amd64/rgb_unseen_test/train/irf'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def process_image(filename):
    try:
        # Construct full paths
        rgb_path = os.path.join(rgb_dir, filename)
        ir_path = os.path.join(ir_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Skip if IR file doesn't exist
        if not os.path.exists(ir_path):
            return f"Warning: IR image not found for {filename}, skipping."
        
        # Load RGB image and convert to grayscale
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            return f"Error: Failed to load RGB image {filename}"
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        
        # Load IR image
        ir_img = cv2.imread(ir_path)
        if ir_img is None:
            return f"Error: Failed to load IR image {filename}"
        
        # Ensure images have same dimensions
        if rgb_img.shape[:2] != ir_img.shape[:2]:
            return f"Error: Image dimensions mismatch for {filename}"
        
        # Create a new 3-channel image with same size as RGB
        merged_img = np.zeros_like(rgb_img)
        
        # Find non-black pixels in IR image (sum across channels)
        ir_mask = (ir_img.sum(axis=2) > 0)
        
        # Create the merged image:
        # - Where IR has data (non-black), use IR values
        # - Where IR is black, use grayscale values (replicated to 3 channels)
        for c in range(3):
            merged_img[:,:,c] = np.where(ir_mask, ir_img[:,:,c], gray_img)
        
        # Save the merged image with original filename
        if not cv2.imwrite(output_path, merged_img):
            return f"Error: Failed to save {filename}"
            
        return f"Success: Processed {filename}"
        
    except Exception as e:
        return f"Error processing {filename}: {str(e)}\n{traceback.format_exc()}"

def main():
    # Get list of files
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Determine number of processes to use
    num_processes = min(cpu_count(), 40)  # Limit to 8 processes to avoid memory issues
    
    print(f"Starting processing with {num_processes} processes...")
    
    # Process images in parallel
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_image, rgb_files), 
                          total=len(rgb_files), 
                          desc="Processing images"))
    
    # Print summary of errors
    errors = [r for r in results if r.startswith('Error') or r.startswith('Warning')]
    if errors:
        print("\nEncountered some issues:")
        for error in errors[:10]:  # Print first 10 errors to avoid flooding
            print(error)
        if len(errors) > 10:
            print(f"... and {len(errors)-10} more issues")
    
    success_count = len([r for r in results if r.startswith('Success')])
    print(f"\nProcessing complete. Successfully processed {success_count}/{len(rgb_files)} images.")

if __name__ == '__main__':
    main()