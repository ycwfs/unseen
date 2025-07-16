#!/usr/bin/env python3
"""
图像检测数据集类别分布分析和增强脚本
用于分析类别分布不均的问题，并通过复制小目标来增强数据集
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import shutil
from pathlib import Path
import json

# 类别定义
CLASSES = {
    0: 'person',
    1: 'cyclist', 
    2: 'car',
    3: 'truck',
    4: 'bus'
}

class DatasetAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.train_images_path = self.dataset_path / 'train' / 'images'
        self.train_labels_path = self.dataset_path / 'train' / 'labels'
        self.class_counts = defaultdict(int)
        self.bbox_sizes = defaultdict(list)  # 存储每个类别的bbox大小
        
    def analyze_class_distribution(self):
        """分析类别分布"""
        print("正在分析类别分布...")
        
        # 遍历所有标签文件
        for label_file in self.train_labels_path.glob('*.txt'):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # 计算bbox面积 (归一化坐标)
                        width = float(parts[3])
                        height = float(parts[4])
                        area = width * height
                        
                        self.class_counts[class_id] += 1
                        self.bbox_sizes[class_id].append(area)
        
        return self.class_counts, self.bbox_sizes
    
    def plot_distribution(self):
        """绘制类别分布图"""
        class_names = [CLASSES[i] for i in sorted(self.class_counts.keys())]
        counts = [self.class_counts[i] for i in sorted(self.class_counts.keys())]
        
        plt.figure(figsize=(12, 5))
        
        # 类别数量分布
        plt.subplot(1, 2, 1)
        bars = plt.bar(class_names, counts, color=['skyblue', 'lightgreen', 'lightcoral', 'orange', 'plum'])
        plt.title('classes distribution')
        plt.xlabel('class')
        plt.ylabel('number')
        plt.xticks(rotation=45)
        
        # 在柱子上显示数值
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom')
        
        # 平均bbox大小分布
        plt.subplot(1, 2, 2)
        avg_sizes = [np.mean(self.bbox_sizes[i]) if self.bbox_sizes[i] else 0 
                    for i in sorted(self.bbox_sizes.keys())]
        bars = plt.bar(class_names, avg_sizes, color=['skyblue', 'lightgreen', 'lightcoral', 'orange', 'plum'])
        plt.title('average object size (normalized area)')
        plt.xlabel('class')
        plt.ylabel('average area')
        plt.xticks(rotation=45)
        
        # 在柱子上显示数值
        for bar, size in zip(bars, avg_sizes):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_sizes)*0.01,
                    f'{size:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('/data1/wangqiurui/code/competition/tianchi/unseen/class_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def identify_minority_classes(self, threshold_ratio=0.3):
        """识别少数类别（需要增强的类别）"""
        total_count = sum(self.class_counts.values())
        avg_count = total_count / len(self.class_counts)

        minority_classes = [0, 1]
        # for class_id, count in self.class_counts.items():
        #     if count < avg_count * threshold_ratio:
        #         minority_classes.append(class_id)
        
        print(f"识别到需要增强的少数类别: {[CLASSES[c] for c in minority_classes]}")
        return minority_classes

class DataAugmenter:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.train_images_path = self.dataset_path / 'train' / 'images'
        self.train_labels_path = self.dataset_path / 'train' / 'labels'
        
        # 创建输出目录
        self.output_images_path = self.output_path / 'train' / 'images'
        self.output_labels_path = self.output_path / 'train' / 'labels'
        self.output_images_path.mkdir(parents=True, exist_ok=True)
        self.output_labels_path.mkdir(parents=True, exist_ok=True)
        
    def is_in_center_region(self, cx, cy, img_width, img_height):
        """检查是否在图像中心256x192区域内"""
        center_x = img_width / 2
        center_y = img_height / 2
        
        # 256x192区域的边界（归一化坐标）
        region_width = 256 / img_width
        region_height = 192 / img_height
        
        left = center_x / img_width - region_width / 2
        right = center_x / img_width + region_width / 2
        top = center_y / img_height - region_height / 2
        bottom = center_y / img_height + region_height / 2
        
        return left <= cx <= right and top <= cy <= bottom
    
    def get_bbox_overlap(self, bbox1, bbox2):
        """计算两个bbox的重叠面积比例"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        
        return inter_area / bbox1_area if bbox1_area > 0 else 0.0
    
    def find_valid_positions(self, img_width, img_height, target_bbox, existing_bboxes, 
                           max_overlap=0.3, min_copies=3, max_copies=6):
        """寻找有效的复制位置"""
        valid_positions = []
        attempts = 0
        max_attempts = 100
        
        target_w = target_bbox[2] - target_bbox[0]
        target_h = target_bbox[3] - target_bbox[1]
        
        while len(valid_positions) < max_copies and attempts < max_attempts:
            attempts += 1
            
            # 随机生成新位置
            new_x1 = random.uniform(0, 1 - target_w)
            new_y1 = random.uniform(0, 1 - target_h)
            new_x2 = new_x1 + target_w
            new_y2 = new_y1 + target_h
            
            new_bbox = [new_x1, new_y1, new_x2, new_y2]
            new_cx = (new_x1 + new_x2) / 2
            new_cy = (new_y1 + new_y2) / 2
            
            # 检查是否在中心区域（要避免）
            if self.is_in_center_region(new_cx, new_cy, img_width, img_height):
                continue
            
            # 检查与现有bbox的重叠
            valid = True
            for existing_bbox in existing_bboxes:
                overlap = self.get_bbox_overlap(new_bbox, existing_bbox)
                if overlap > max_overlap:
                    valid = False
                    break
            
            if valid:
                valid_positions.append([new_cx, new_cy, target_w, target_h])
        
        # 至少保证有min_copies个副本
        if len(valid_positions) < min_copies:
            # 放宽条件再试一次
            while len(valid_positions) < min_copies and attempts < max_attempts * 2:
                attempts += 1
                new_x1 = random.uniform(0, 1 - target_w)
                new_y1 = random.uniform(0, 1 - target_h)
                new_x2 = new_x1 + target_w
                new_y2 = new_y1 + target_h
                
                new_bbox = [new_x1, new_y1, new_x2, new_y2]
                new_cx = (new_x1 + new_x2) / 2
                new_cy = (new_y1 + new_y2) / 2
                
                if not self.is_in_center_region(new_cx, new_cy, img_width, img_height):
                    valid_positions.append([new_cx, new_cy, target_w, target_h])
        
        return valid_positions[:max_copies]
    
    def augment_image(self, img_path, label_path, minority_classes, copy_factor=3):
        """增强单张图像"""
        # 读取图像
        image = cv2.imread(str(img_path))
        if image is None:
            return False
        
        img_height, img_width = image.shape[:2]
        
        # 读取标签
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        annotations = []
        minority_targets = []
        
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    
                    annotation = [class_id, cx, cy, w, h]
                    annotations.append(annotation)
                    
                    # 收集少数类别的目标
                    if class_id in minority_classes:
                        # 将归一化坐标转换为像素坐标用于复制
                        x1 = int((cx - w/2) * img_width)
                        y1 = int((cy - h/2) * img_height)
                        x2 = int((cx + w/2) * img_width)
                        y2 = int((cy + h/2) * img_height)
                        
                        # 确保坐标在图像范围内
                        x1 = max(0, min(x1, img_width-1))
                        y1 = max(0, min(y1, img_height-1))
                        x2 = max(0, min(x2, img_width-1))
                        y2 = max(0, min(y2, img_height-1))
                        
                        if x2 > x1 and y2 > y1:
                            minority_targets.append({
                                'class_id': class_id,
                                'region': image[y1:y2, x1:x2],
                                'bbox_norm': [cx-w/2, cy-h/2, cx+w/2, cy+h/2],
                                'size_norm': [w, h]
                            })
        
        if not minority_targets:
            return False
        
        # 复制图像
        augmented_image = image.copy()
        new_annotations = annotations.copy()
        
        # 现有的bbox（用于避免重叠）
        existing_bboxes = [[cx-w/2, cy-h/2, cx+w/2, cy+h/2] for _, cx, cy, w, h in annotations]
        
        # 为每个少数类别目标创建副本
        for target in minority_targets:
            target_region = target['region']
            class_id = target['class_id']
            w_norm, h_norm = target['size_norm']
            
            # 寻找有效位置
            valid_positions = self.find_valid_positions(
                img_width, img_height, target['bbox_norm'], 
                existing_bboxes, max_copies=copy_factor
            )
            
            for pos in valid_positions:
                new_cx, new_cy, new_w, new_h = pos
                
                # 转换为像素坐标
                new_x1 = int((new_cx - new_w/2) * img_width)
                new_y1 = int((new_cy - new_h/2) * img_height)
                new_x2 = int((new_cx + new_w/2) * img_width)
                new_y2 = int((new_cy + new_h/2) * img_height)
                
                # 确保坐标在图像范围内
                new_x1 = max(0, min(new_x1, img_width-1))
                new_y1 = max(0, min(new_y1, img_height-1))
                new_x2 = max(1, min(new_x2, img_width))
                new_y2 = max(1, min(new_y2, img_height))
                
                # 调整目标区域大小到新位置
                new_width = new_x2 - new_x1
                new_height = new_y2 - new_y1
                
                if new_width > 0 and new_height > 0:
                    resized_region = cv2.resize(target_region, (new_width, new_height))
                    
                    # 复制到新位置
                    augmented_image[new_y1:new_y2, new_x1:new_x2] = resized_region
                    
                    # 添加新的标注
                    new_annotations.append([class_id, new_cx, new_cy, new_w, new_h])
                    existing_bboxes.append([new_cx-new_w/2, new_cy-new_h/2, new_cx+new_w/2, new_cy+new_h/2])
        
        return augmented_image, new_annotations
    
    def augment_dataset(self, minority_classes, copy_factor=3):
        """增强整个数据集"""
        print(f"开始增强数据集，目标类别: {[CLASSES[c] for c in minority_classes]}")
        
        # 首先复制所有原始文件
        print("复制原始文件...")
        for img_file in self.train_images_path.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy2(img_file, self.output_images_path)
        
        for label_file in self.train_labels_path.glob('*.txt'):
            shutil.copy2(label_file, self.output_labels_path)
        
        # 增强包含少数类别的图像
        augmented_count = 0
        processed_count = 0
        
        for img_file in self.train_images_path.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                processed_count += 1
                
                label_file = self.train_labels_path / (img_file.stem + '.txt')
                if not label_file.exists():
                    continue
                
                # 检查是否包含少数类别
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                has_minority = False
                for line in lines:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if class_id in minority_classes:
                                has_minority = True
                                break
                
                if has_minority:
                    result = self.augment_image(img_file, label_file, minority_classes, copy_factor)
                    if result:
                        augmented_image, new_annotations = result
                        
                        # 保存增强的图像
                        # output_img_name = f"aug_{img_file.name}"
                        # 覆盖原图像
                        output_img_name = img_file.name
                        output_img_path = self.output_images_path / output_img_name
                        print(f"保存增强的图像: {output_img_path}")
                        cv2.imwrite(str(output_img_path), augmented_image)
                        
                        # 保存新的标注
                        # output_label_path = self.output_labels_path / f"aug_{img_file.stem}.txt"
                        # 覆盖原标注
                        output_label_path = self.output_labels_path / f"{img_file.stem}.txt"
                        with open(output_label_path, 'w') as f:
                            for ann in new_annotations:
                                f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
                        
                        augmented_count += 1
                
                if processed_count % 100 == 0:
                    print(f"已处理 {processed_count} 张图像，增强了 {augmented_count} 张")
        
        print(f"增强完成！总共处理 {processed_count} 张图像，增强了 {augmented_count} 张")
        
        return augmented_count

def main():
    dataset_path = '/data1/wangqiurui/code/ossutil-v1.7.19-linux-amd64/rgb_unseen'
    output_path = '/data1/wangqiurui/code/ossutil-v1.7.19-linux-amd64/rgb_unseen_augmented'
    
    print("=== 数据集类别分布分析 ===")
    
    # 分析类别分布
    analyzer = DatasetAnalyzer(dataset_path)
    class_counts, bbox_sizes = analyzer.analyze_class_distribution()
    
    print("\n类别分布统计:")
    for class_id in sorted(class_counts.keys()):
        class_name = CLASSES[class_id]
        count = class_counts[class_id]
        avg_size = np.mean(bbox_sizes[class_id]) if bbox_sizes[class_id] else 0
        print(f"{class_name}: {count} 个目标, 平均大小: {avg_size:.4f}")
    
    # 绘制分布图
    analyzer.plot_distribution()
    
    # 识别需要增强的少数类别
    minority_classes = analyzer.identify_minority_classes(threshold_ratio=0.5)
    
    if not minority_classes:
        print("没有发现需要增强的少数类别")
        return
    
    print(f"\n=== 开始数据增强 ===")
    print(f"将对以下类别进行增强: {[CLASSES[c] for c in minority_classes]}")
    
    # 进行数据增强
    augmenter = DataAugmenter(dataset_path, output_path)
    augmented_count = augmenter.augment_dataset(minority_classes, copy_factor=4)
    
    print(f"\n=== 增强完成 ===")
    print(f"增强后的数据集保存在: {output_path}")
    print(f"增强了 {augmented_count} 张图像")

if __name__ == "__main__":
    main()
