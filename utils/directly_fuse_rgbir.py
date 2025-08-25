import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import glob

def convert_to_color_image(fi, cr_channel, cb_channel):
    """
    将融合图像和YCrCb通道数据转换为RGB彩色图像
    """
    # 确保图像数据为uint8格式
    fi = fi.astype(np.uint8)
 
    # 组合YCrCb通道
    ycrcb_fi = np.dstack((fi, cr_channel, cb_channel))
 
    # 将YCrCb图像转换为RGB
    rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
 
    return rgb_fi

def process_single_image(ir_path, rgb_path, save_path):
    """
    处理单个图像对的融合
    """
    try:
        # 读取融合图像和可见光图像
        fused_image = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)  # 融合图像（灰度）
        vis_image = cv2.imread(rgb_path)  # 可见光图像
        
        if fused_image is None:
            print(f"Error: Could not read IR image {ir_path}")
            return False
        
        if vis_image is None:
            print(f"Error: Could not read RGB image {rgb_path}")
            return False
        
        # 分离可见光图像的 Cr 和 Cb 通道
        vis_ycrcb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2YCrCb)
        _, cr_channel, cb_channel = cv2.split(vis_ycrcb)
        
        # 转换为彩色图像
        color_image = convert_to_color_image(fused_image, cr_channel, cb_channel)
        
        # 保存图像
        success = cv2.imwrite(save_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
        
        if success:
            print(f"Successfully processed: {os.path.basename(ir_path)}")
            return True
        else:
            print(f"Error: Failed to save {save_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {ir_path}: {str(e)}")
        return False

def batch_process_images(ir_folder, rgb_folder, save_folder, max_workers=4):
    """
    批量并行处理图像融合
    """
    # 创建保存目录
    os.makedirs(save_folder, exist_ok=True)
    
    # 获取所有IR图像文件
    ir_files = glob.glob(os.path.join(ir_folder, "*.jpg")) + \
               glob.glob(os.path.join(ir_folder, "*.png")) + \
               glob.glob(os.path.join(ir_folder, "*.jpeg"))
    
    print(f"Found {len(ir_files)} IR images to process")
    
    # 准备任务列表
    tasks = []
    for ir_path in ir_files:
        filename = os.path.basename(ir_path)
        rgb_path = os.path.join(rgb_folder, filename)
        save_path = os.path.join(save_folder, filename)
        
        # 检查对应的RGB图像是否存在
        if os.path.exists(rgb_path):
            tasks.append((ir_path, rgb_path, save_path))
        else:
            print(f"Warning: RGB image not found for {filename}")
    
    print(f"Processing {len(tasks)} image pairs...")
    
    # 使用线程池并行处理
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(process_single_image, ir_path, rgb_path, save_path): (ir_path, rgb_path, save_path)
            for ir_path, rgb_path, save_path in tasks
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_task):
            ir_path, rgb_path, save_path = future_to_task[future]
            try:
                result = future.result()
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Exception in processing {os.path.basename(ir_path)}: {str(e)}")
                failed += 1
    
    print(f"\nProcessing completed!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(tasks)}")

# 主程序
if __name__ == "__main__":
    # 设置路径
    base_path = '/data1/code/ossutil-v1.7.19-linux-amd64/rgb_unseen_test/train'
    ir_folder = os.path.join(base_path, 'irf')
    rgb_folder = os.path.join(base_path, 'imageso')
    save_folder = os.path.join(base_path, 'images')
    
    # 检查文件夹是否存在
    if not os.path.exists(ir_folder):
        print(f"Error: IR folder does not exist: {ir_folder}")
        exit(1)
    
    if not os.path.exists(rgb_folder):
        print(f"Error: RGB folder does not exist: {rgb_folder}")
        exit(1)
    
    # 开始批量处理
    batch_process_images(ir_folder, rgb_folder, save_folder, max_workers=40)