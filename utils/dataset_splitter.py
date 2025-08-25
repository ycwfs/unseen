#!/usr/bin/env python3
"""
智能数据集划分脚本
根据图像中物体类别分布，合理切分训练集和验证集
确保验证集中各类别的分布与训练集保持一致
"""

import os
import shutil
import random
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime

# 类别定义
CLASSES = {
    0: 'person',
    1: 'cyclist', 
    2: 'car',
    3: 'truck',
    4: 'bus'
}

class DatasetSplitter:
    def __init__(self, dataset_path, val_ratio=0.2):
        self.dataset_path = Path(dataset_path)
        self.val_ratio = val_ratio
        self.train_images_path = self.dataset_path / 'train' / 'images'
        self.train_labels_path = self.dataset_path / 'train' / 'labels'
        
        # 统计数据
        self.image_class_info = {}  # 每张图像的类别信息
        self.class_counts = defaultdict(int)  # 各类别总数
        self.images_by_class_count = defaultdict(list)  # 按类别数量分组的图像
        
    def analyze_dataset(self):
        """分析数据集的类别分布"""
        print("正在分析数据集...")
        
        image_files = list(self.train_images_path.glob('*.jpg'))
        total_images = len(image_files)
        print(f"发现 {total_images} 张图像")
        
        for i, img_file in enumerate(image_files):
            if i % 1000 == 0:
                print(f"已分析 {i}/{total_images} 张图像")
                
            img_name = img_file.stem
            label_file = self.train_labels_path / f"{img_name}.txt"
            
            if label_file.exists():
                # 读取标签文件
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                image_classes = set()
                class_counts_in_image = defaultdict(int)
                
                for line in lines:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            image_classes.add(class_id)
                            class_counts_in_image[class_id] += 1
                            self.class_counts[class_id] += 1
                
                # 存储图像信息
                self.image_class_info[img_name] = {
                    'classes': list(image_classes),
                    'class_counts': dict(class_counts_in_image),
                    'num_classes': len(image_classes),
                    'total_objects': sum(class_counts_in_image.values())
                }
                
                # 按类别数量分组
                num_classes = len(image_classes)
                self.images_by_class_count[num_classes].append(img_name)
        
        print("数据集分析完成！")
        return self.image_class_info, self.class_counts
    
    def print_analysis_summary(self):
        """打印分析摘要"""
        print("\n=== 数据集分析摘要 ===")
        
        # 总体统计
        total_images = len(self.image_class_info)
        total_objects = sum(self.class_counts.values())
        
        print(f"总图像数: {total_images}")
        print(f"总目标数: {total_objects}")
        print(f"平均每张图像目标数: {total_objects/total_images:.2f}")
        
        # 各类别分布
        print("\n类别分布:")
        for class_id in sorted(self.class_counts.keys()):
            class_name = CLASSES[class_id]
            count = self.class_counts[class_id]
            percentage = count / total_objects * 100
            print(f"{class_name}: {count} ({percentage:.2f}%)")
        
        # 按类别数量分组统计
        print("\n按图像中类别数量分组:")
        for num_classes in sorted(self.images_by_class_count.keys()):
            count = len(self.images_by_class_count[num_classes])
            percentage = count / total_images * 100
            print(f"{num_classes} 个类别: {count} 张图像 ({percentage:.2f}%)")
    
    def stratified_split(self):
        """基于类别分布的分层采样"""
        print(f"\n开始分层采样，验证集比例: {self.val_ratio}")
        
        train_images = []
        val_images = []
        
        # 为了保证各类别在验证集中都有足够的代表性
        # 我们需要考虑以下策略：
        # 1. 保证每个类别在验证集中都有一定数量的样本
        # 2. 保证包含多个类别的图像也能在验证集中体现
        # 3. 保持类别分布的平衡
        
        # 先为每个类别收集包含该类别的图像
        images_with_class = defaultdict(list)
        for img_name, info in self.image_class_info.items():
            for class_id in info['classes']:
                images_with_class[class_id].append(img_name)
        
        # 计算每个类别需要在验证集中的最小数量
        min_val_per_class = max(10, int(self.val_ratio * 1000))  # 至少10张或按比例
        
        # 使用多步策略进行采样
        selected_for_val = set()
        
        # 第一步：为每个类别保证最少的验证样本
        for class_id in sorted(CLASSES.keys()):
            if class_id in images_with_class:
                class_images = images_with_class[class_id]
                random.shuffle(class_images)
                
                # 选择包含该类别的图像加入验证集
                needed = min_val_per_class
                added = 0
                
                for img_name in class_images:
                    if img_name not in selected_for_val and added < needed:
                        selected_for_val.add(img_name)
                        added += 1
        
        # 第二步：随机补充到目标数量
        all_images = list(self.image_class_info.keys())
        random.shuffle(all_images)
        
        target_val_count = int(len(all_images) * self.val_ratio)
        
        for img_name in all_images:
            if len(selected_for_val) >= target_val_count:
                break
            if img_name not in selected_for_val:
                selected_for_val.add(img_name)
        
        # 分配剩余图像到训练集
        for img_name in all_images:
            if img_name in selected_for_val:
                val_images.append(img_name)
            else:
                train_images.append(img_name)
        
        print(f"训练集: {len(train_images)} 张图像")
        print(f"验证集: {len(val_images)} 张图像")
        print(f"实际验证集比例: {len(val_images)/len(all_images):.2%}")
        
        return train_images, val_images
    
    def validate_split(self, train_images, val_images):
        """验证划分的质量"""
        print("\n=== 验证划分质量 ===")
        
        # 统计训练集和验证集的类别分布
        train_class_counts = defaultdict(int)
        val_class_counts = defaultdict(int)
        
        for img_name in train_images:
            if img_name in self.image_class_info:
                for class_id, count in self.image_class_info[img_name]['class_counts'].items():
                    train_class_counts[class_id] += count
        
        for img_name in val_images:
            if img_name in self.image_class_info:
                for class_id, count in self.image_class_info[img_name]['class_counts'].items():
                    val_class_counts[class_id] += count
        
        print("类别分布对比:")
        print(f"{'类别':<10} {'训练集':<10} {'验证集':<10} {'训练集%':<10} {'验证集%':<10}")
        print("-" * 60)
        
        total_train = sum(train_class_counts.values())
        total_val = sum(val_class_counts.values())
        
        for class_id in sorted(CLASSES.keys()):
            class_name = CLASSES[class_id]
            train_count = train_class_counts[class_id]
            val_count = val_class_counts[class_id]
            train_pct = train_count / total_train * 100 if total_train > 0 else 0
            val_pct = val_count / total_val * 100 if total_val > 0 else 0
            
            print(f"{class_name:<10} {train_count:<10} {val_count:<10} {train_pct:<10.2f} {val_pct:<10.2f}")
        
        return train_class_counts, val_class_counts
    
    def backup_original_data(self):
        """备份原始数据"""
        print("\n正在备份原始数据...")
        
        backup_dir = self.dataset_path / f"train_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(exist_ok=True)
        
        # 备份train文件夹
        train_backup = backup_dir / "train"
        if self.dataset_path / "train" != train_backup:
            shutil.copytree(self.dataset_path / "train", train_backup)
        
        print(f"原始数据已备份到: {backup_dir}")
        return backup_dir
    
    def create_new_split(self, train_images, val_images, backup_dir):
        """创建新的数据集划分"""
        print("\n正在创建新的数据集划分...")
        
        # 创建新的目录结构
        new_train_images = self.dataset_path / "train_new" / "images"
        new_train_labels = self.dataset_path / "train_new" / "labels"
        new_train_irs = self.dataset_path / "train_new" / "ir"
        new_val_images = self.dataset_path / "val_new" / "images"
        new_val_labels = self.dataset_path / "val_new" / "labels"
        new_val_irs = self.dataset_path / "val_new" / "ir"
        
        # 清空并重新创建目录
        for dir_path in [new_train_images, new_train_labels, new_train_irs, new_val_images, new_val_labels, new_val_irs]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 复制训练集文件
        print("复制训练集文件...")
        for img_name in train_images:
            # 复制图像
            src_img = backup_dir / "train" / "images" / f"{img_name}.jpg"
            dst_img = new_train_images / f"{img_name}.jpg"
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            
            # 复制标签
            src_label = backup_dir / "train" / "labels" / f"{img_name}.txt"
            dst_label = new_train_labels / f"{img_name}.txt"
            if src_label.exists():
                shutil.copy2(src_label, dst_label)

            # 复制红外图像
            src_ir = backup_dir / "train" / "ir" / f"{img_name}.jpg"
            dst_ir = new_train_irs / f"{img_name}.jpg"
            if src_label.exists():
                shutil.copy2(src_ir, dst_ir)            
        
        # 复制验证集文件
        print("复制验证集文件...")
        for img_name in val_images:
            # 复制图像
            src_img = backup_dir / "train" / "images" / f"{img_name}.jpg"
            dst_img = new_val_images / f"{img_name}.jpg"
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            
            # 复制标签
            src_label = backup_dir / "train" / "labels" / f"{img_name}.txt"
            dst_label = new_val_labels / f"{img_name}.txt"
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
            
            # 复制红外图像
            src_ir = backup_dir / "train" / "ir" / f"{img_name}.jpg"
            dst_ir = new_val_irs / f"{img_name}.jpg"
            if src_label.exists():
                shutil.copy2(src_ir, dst_ir)             
        
        print(f"新训练集: {len(train_images)} 张图像")
        print(f"新验证集: {len(val_images)} 张图像")
    
    def save_split_info(self, train_images, val_images, train_class_counts, val_class_counts):
        """保存划分信息"""
        split_info = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(train_images) + len(val_images),
            'train_images': len(train_images),
            'val_images': len(val_images),
            'val_ratio': len(val_images) / (len(train_images) + len(val_images)),
            'train_class_distribution': dict(train_class_counts),
            'val_class_distribution': dict(val_class_counts),
            'classes': CLASSES
        }
        
        info_file = self.dataset_path / "split_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        
        print(f"划分信息已保存到: {info_file}")
    
    def plot_distribution_comparison(self, train_class_counts, val_class_counts):
        """绘制类别分布对比图"""
        class_names = [CLASSES[i] for i in sorted(CLASSES.keys())]
        train_counts = [train_class_counts[i] for i in sorted(CLASSES.keys())]
        val_counts = [val_class_counts[i] for i in sorted(CLASSES.keys())]
        
        x = np.arange(len(class_names))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绝对数量对比
        ax1.bar(x - width/2, train_counts, width, label='train', alpha=0.8)
        ax1.bar(x + width/2, val_counts, width, label='val', alpha=0.8)
        ax1.set_xlabel('class')
        ax1.set_ylabel('number')
        ax1.set_title('Train vs Val - abs')
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 相对比例对比
        total_train = sum(train_counts)
        total_val = sum(val_counts)
        train_pcts = [count/total_train*100 for count in train_counts]
        val_pcts = [count/total_val*100 for count in val_counts]
        
        ax2.bar(x - width/2, train_pcts, width, label='train', alpha=0.8)
        ax2.bar(x + width/2, val_pcts, width, label='val', alpha=0.8)
        ax2.set_xlabel('class')
        ax2.set_ylabel('percentage (%)')
        ax2.set_title('train vs Val - revelent %')
        ax2.set_xticks(x)
        ax2.set_xticklabels(class_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = self.dataset_path / "split_distribution_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"分布对比图已保存到: {plot_file}")
        plt.close()
    
    def run_split(self):
        """执行完整的数据集划分流程"""
        print("=== 开始智能数据集划分 ===")
        
        # 设置随机种子
        random.seed(42)
        np.random.seed(42)
        
        # 1. 分析数据集
        self.analyze_dataset()
        self.print_analysis_summary()
        
        # 2. 备份原始数据
        backup_dir = self.backup_original_data()
        
        # 3. 执行分层采样
        train_images, val_images = self.stratified_split()
        
        # 4. 验证划分质量
        train_class_counts, val_class_counts = self.validate_split(train_images, val_images)
        
        # 5. 创建新的数据集划分
        self.create_new_split(train_images, val_images, backup_dir)
        
        # 6. 保存划分信息
        self.save_split_info(train_images, val_images, train_class_counts, val_class_counts)
        
        # 7. 绘制分布对比图
        self.plot_distribution_comparison(train_class_counts, val_class_counts)
        
        print("\n=== 数据集划分完成！ ===")
        print(f"备份目录: {backup_dir}")
        print(f"新训练集: {len(train_images)} 张图像")
        print(f"新验证集: {len(val_images)} 张图像")

def main():
    dataset_path = '/data1/code/ossutil-v1.7.19-linux-amd64/rgb_unseen'
    val_ratio = 0.2  # 验证集比例
    
    print(f"数据集路径: {dataset_path}")
    print(f"验证集比例: {val_ratio}")
    
    # 检查数据集路径
    if not Path(dataset_path).exists():
        print(f"错误: 数据集路径不存在: {dataset_path}")
        return
    
    if not (Path(dataset_path) / 'train' / 'images').exists():
        print(f"错误: 训练图像路径不存在: {dataset_path}/train/images")
        return
    
    if not (Path(dataset_path) / 'train' / 'labels').exists():
        print(f"错误: 训练标签路径不存在: {dataset_path}/train/labels")
        return
    
    # 创建划分器并执行
    splitter = DatasetSplitter(dataset_path, val_ratio)
    splitter.run_split()

if __name__ == "__main__":
    main()
