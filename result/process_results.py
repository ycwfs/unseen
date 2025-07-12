import os

# Get the absolute path to the directory containing this script
res_dir = os.path.dirname(os.path.abspath(__file__))
pred_dir = "/data1/wangqiurui/code/competition/tianchi/unseen/yolov12/yolov11s_rgb_1280/1280_epoch10_res/labels"
# Define paths for the labels directory and the output file
# labels_dir = os.path.join(pred_dir, "labels")
output_file = os.path.join(res_dir, "result_srgb_yolo11_89_1280.txt")

# --- (Optional) Hardcoded model parameters for demonstration ---
# Replace with your actual model's parameter and calculation amount
model_params = "532105 19"  # Example: num_params GFLOPs

# --- Main processing logic ---
with open(output_file, "w") as f_out:
    # Write the header line with model parameters
    f_out.write(f"{model_params}\n")

    # Get all .txt files from the labels directory
    label_files = [f for f in os.listdir(pred_dir) if f.endswith(".txt")]

    num_files = len(label_files)
    # Process each label file
    for filename in label_files:
        image_name = os.path.splitext(filename)[0] + ".jpg"
        
        # Initialize a list to store detection data for the current image
        detections = []

        with open(os.path.join(pred_dir, filename), "r") as f_in:
            for line in f_in:
                parts = line.strip().split()
                
                # Ensure there are enough parts to represent a detection
                if len(parts) == 6:
                    # Extract detection attributes
                    class_id, x_center, y_center, width, height, confidence = map(float, parts)

                    # Filter detections by confidence score
                    if confidence > 0.25:
                        # Append relevant detection data to the list
                        detections.extend([x_center, y_center, width, height, confidence, int(class_id)])

        # If there are valid detections, write them to the output file
        if detections:
            # Format the output line with the image name followed by detection data
            output_line = f"{image_name} {' '.join(map(str, detections))}\n"
            f_out.write(output_line)
    
    nn = 0
    # process none detections
    for i in range(5000, 15000):
        # Format: 005000 005001 005002 ... 014999
        if i < 10000:
            filename = f'00{i}.txt'
        else:
            filename = f'0{i}.txt'
            
        if os.path.exists(os.path.join(pred_dir, filename)):
            continue
        else:
            image_name = os.path.splitext(filename)[0] + ".jpg"
            f_out.write(f"{image_name}\n")
            print(f"{image_name}")
            nn += 1


print(f"Processing complete. {num_files} + {nn} Results are saved in {output_file}")
