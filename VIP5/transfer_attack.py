#!/usr/bin/env python3
"""
VIP5 黑盒对抗攻击 - Transfer Attack
====================================
基于CLIP的转移攻击方法

原理：
    使用CLIP模型的梯度生成对抗扰动，由于VIP5也使用CLIP提取视觉特征，
    因此在CLIP上生成的对抗扰动可以"转移"到VIP5模型上产生效果。
    
    这是一种黑盒攻击，因为：
    - 我们不需要访问VIP5模型的内部结构
    - 我们不需要VIP5模型的梯度
    - 我们只利用CLIP作为代理模型

使用方法：
    python transfer_attack.py --split toys --num_images 100
    python transfer_attack.py --split toys --num_images 100 --epsilon 0.08 --max_iter 30

作者: 对抗攻击研究
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import random
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# 导入CLIP
import clip

SCRIPT_DIR = Path(__file__).resolve().parent


def pil_to_tensor(img: Image.Image, device='cuda') -> torch.Tensor:
    """将PIL图像转换为tensor (0-1范围)"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0).to(device)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """将tensor转换回PIL图像"""
    tensor = tensor.squeeze(0).cpu()
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)


def normalize_for_clip(tensor: torch.Tensor) -> torch.Tensor:
    """CLIP标准化"""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(tensor.device)
    return (tensor - mean) / std


class TransferAttacker:
    """
    基于CLIP的转移攻击器
    
    攻击策略：
    1. maximize_change: 最大化特征变化（让图像特征尽可能偏离原始特征）
    2. toward_target: 向目标特征方向移动（让图像特征接近热门商品特征）
    
    参数说明：
    - epsilon: 最大扰动幅度，范围[0,1]，建议0.03-0.1
    - max_iter: PGD迭代次数，建议10-50
    - step_size: 每步更新的步长，建议epsilon/max_iter的2-4倍
    """
    
    def __init__(self, device='cuda', epsilon=0.05, max_iter=20, 
                 step_size=0.005, attack_type='maximize_change'):
        """
        初始化攻击器
        
        Args:
            device: 计算设备 ('cuda' 或 'cpu')
            epsilon: 最大扰动幅度 (0-1)
            max_iter: 迭代次数
            step_size: 每步更新步长
            attack_type: 攻击类型 ('maximize_change' 或 'toward_target')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.step_size = step_size
        self.attack_type = attack_type
        self.target_features = None
        
        # 加载CLIP模型
        print(f"Loading CLIP model on {self.device}...")
        self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device=self.device)
        self.clip_model.eval()
        print("CLIP model loaded!")
    
    def extract_feature(self, image: Image.Image) -> np.ndarray:
        """提取CLIP特征"""
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_image(image_input)
        return features.cpu().numpy().squeeze()
    
    def set_target_features(self, features: np.ndarray):
        """设置目标特征（用于toward_target攻击类型）"""
        self.target_features = torch.from_numpy(features).float().to(self.device)
        print(f"Target features set, shape: {features.shape}")
    
    def attack_single(self, image: Image.Image) -> Image.Image:
        """
        对单张图像执行攻击
        
        Args:
            image: 原始PIL图像
            
        Returns:
            attacked_image: 攻击后的PIL图像
        """
        # 转换为tensor
        img_tensor = pil_to_tensor(image, self.device)
        original_tensor = img_tensor.clone()
        
        # 获取原始特征
        with torch.no_grad():
            original_normalized = normalize_for_clip(img_tensor)
            original_feat = self.clip_model.encode_image(original_normalized)
        
        # 设置目标特征
        if self.attack_type == 'toward_target' and self.target_features is not None:
            target = self.target_features
        else:
            target = None
        
        # PGD迭代攻击
        perturbed = img_tensor.clone()
        
        for i in range(self.max_iter):
            perturbed.requires_grad = True
            
            # 前向传播
            normalized = normalize_for_clip(perturbed)
            features = self.clip_model.encode_image(normalized)
            
            # 计算损失
            if self.attack_type == 'toward_target' and target is not None:
                # 最小化与目标特征的距离（让特征接近热门商品）
                loss = 1 - F.cosine_similarity(features, target.unsqueeze(0), dim=1).mean()
            else:
                # 最大化与原始特征的距离（让特征尽可能变化）
                loss = -F.cosine_similarity(features, original_feat, dim=1).mean()
            
            # 反向传播
            loss.backward()
            
            # PGD更新
            with torch.no_grad():
                # 使用梯度符号更新（FGSM风格）
                grad_sign = perturbed.grad.sign()
                perturbed = perturbed - self.step_size * grad_sign
                
                # 投影到epsilon球内（L∞约束）
                delta = perturbed - original_tensor
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                
                # 确保像素值在有效范围内
                perturbed = torch.clamp(original_tensor + delta, 0, 1)
            
            perturbed = perturbed.detach()
        
        # 转换回PIL图像并恢复原始大小
        attacked_img = tensor_to_pil(perturbed)
        attacked_img = attacked_img.resize(image.size, Image.LANCZOS)
        
        return attacked_img
    
    def attack_batch(self, image_dir: Path, output_dir: Path, 
                     num_images: Optional[int] = None) -> Dict:
        """
        批量攻击图像
        
        Args:
            image_dir: 原始图像目录
            output_dir: 攻击后图像保存目录
            num_images: 攻击图像数量（None表示全部）
            
        Returns:
            results: 包含攻击统计信息的字典
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像文件
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        # 随机采样
        if num_images and num_images < len(image_files):
            random.seed(42)  # 固定随机种子以保证可复现
            image_files = random.sample(image_files, num_images)
        
        print(f"\n{'='*60}")
        print(f"开始攻击 {len(image_files)} 张图像")
        print(f"{'='*60}")
        print(f"攻击类型: {self.attack_type}")
        print(f"Epsilon: {self.epsilon}")
        print(f"迭代次数: {self.max_iter}")
        print(f"步长: {self.step_size}")
        print(f"{'='*60}\n")
        
        results = {
            'success': 0,
            'failed': 0,
            'details': []
        }
        
        for img_path in tqdm(image_files, desc="Attacking"):
            try:
                # 加载原始图像
                original_img = Image.open(img_path).convert('RGB')
                original_feat = self.extract_feature(original_img)
                
                # 执行攻击
                attacked_img = self.attack_single(original_img)
                attacked_feat = self.extract_feature(attacked_img)
                
                # 保存攻击后的图像
                output_path = output_dir / img_path.name
                attacked_img.save(output_path, quality=95)
                
                # 计算特征变化
                feat_diff = np.linalg.norm(attacked_feat - original_feat)
                cosine_sim = np.dot(original_feat, attacked_feat) / (
                    np.linalg.norm(original_feat) * np.linalg.norm(attacked_feat) + 1e-8
                )
                
                results['success'] += 1
                results['details'].append({
                    'image': img_path.name,
                    'feature_diff': float(feat_diff),
                    'cosine_sim': float(cosine_sim)
                })
                
            except Exception as e:
                print(f"\nError processing {img_path.name}: {e}")
                results['failed'] += 1
        
        return results


def get_popular_items(split: str, top_k: int = 50) -> List[str]:
    """
    从sequential_data.txt获取热门商品列表
    
    Args:
        split: 数据集名称 (如 'toys')
        top_k: 返回前k个热门商品
        
    Returns:
        popular_items: 热门商品ID列表
    """
    data_dir = SCRIPT_DIR / 'data' / split
    sequential_path = data_dir / 'sequential_data.txt'
    
    if not sequential_path.exists():
        print(f"Warning: {sequential_path} not found")
        return []
    
    from collections import Counter
    item_counts = Counter()
    
    with open(sequential_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) > 1:
                items = parts[1:]
                for item in items:
                    item_counts[item] += 1
    
    popular = [item for item, _ in item_counts.most_common(top_k)]
    print(f"Found {len(popular)} popular items from sequential data")
    return popular


def load_target_features(feature_dir: Path, popular_items: List[str]) -> Optional[np.ndarray]:
    """
    加载热门商品的平均特征作为攻击目标
    
    Args:
        feature_dir: 特征文件目录
        popular_items: 热门商品ID列表
        
    Returns:
        target_feature: 目标特征向量
    """
    features = []
    
    for item_id in popular_items:
        feat_path = feature_dir / f"{item_id}.npy"
        if feat_path.exists():
            features.append(np.load(feat_path))
    
    if len(features) == 0:
        print("Warning: No target features found")
        return None
    
    target_feature = np.mean(features, axis=0)
    print(f"Loaded target feature from {len(features)} popular items")
    return target_feature


def main():
    parser = argparse.ArgumentParser(
        description='VIP5 黑盒对抗攻击 - Transfer Attack',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法（最大化特征变化）
  python transfer_attack.py --split toys --num_images 100
  
  # 向热门商品特征靠近
  python transfer_attack.py --split toys --num_images 100 --attack_type toward_target
  
  # 调整扰动强度
  python transfer_attack.py --split toys --num_images 100 --epsilon 0.08 --max_iter 30

参数建议:
  - epsilon: 0.03-0.1 (越大扰动越明显，但可能影响图像质量)
  - max_iter: 10-50 (越多效果越好，但速度越慢)
  - step_size: 建议设为 epsilon/max_iter 的 2-4 倍
        """
    )
    
    parser.add_argument('--split', type=str, default='toys',
                        help='数据集名称 (默认: toys)')
    parser.add_argument('--num_images', type=int, default=None,
                        help='攻击图像数量 (默认: 全部)')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='最大扰动幅度 (默认: 0.05)')
    parser.add_argument('--max_iter', type=int, default=20,
                        help='PGD迭代次数 (默认: 20)')
    parser.add_argument('--step_size', type=float, default=0.005,
                        help='每步更新步长 (默认: 0.005)')
    parser.add_argument('--attack_type', type=str, default='maximize_change',
                        choices=['maximize_change', 'toward_target'],
                        help='攻击类型 (默认: maximize_change)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (默认: cuda)')
    
    args = parser.parse_args()
    
    # 设置路径
    image_dir = SCRIPT_DIR / args.split
    output_dir = SCRIPT_DIR / f'{args.split}2'
    feature_dir = SCRIPT_DIR / 'features' / 'vitb32_features' / f'{args.split}_original'
    
    # 检查输入目录
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("VIP5 黑盒对抗攻击 - Transfer Attack")
    print("="*60)
    print(f"输入目录: {image_dir}")
    print(f"输出目录: {output_dir}")
    print(f"攻击类型: {args.attack_type}")
    print(f"Epsilon: {args.epsilon}")
    print(f"迭代次数: {args.max_iter}")
    print(f"步长: {args.step_size}")
    
    # 创建攻击器
    attacker = TransferAttacker(
        device=args.device,
        epsilon=args.epsilon,
        max_iter=args.max_iter,
        step_size=args.step_size,
        attack_type=args.attack_type
    )
    
    # 如果是toward_target类型，加载目标特征
    if args.attack_type == 'toward_target':
        if feature_dir.exists():
            popular_items = get_popular_items(args.split)
            if popular_items:
                target_feat = load_target_features(feature_dir, popular_items[:20])
                if target_feat is not None:
                    attacker.set_target_features(target_feat)
        else:
            print(f"Warning: Feature directory not found: {feature_dir}")
            print("Falling back to maximize_change attack type")
            attacker.attack_type = 'maximize_change'
    
    # 执行攻击
    results = attacker.attack_batch(image_dir, output_dir, args.num_images)
    
    # 打印结果
    print("\n" + "="*60)
    print("攻击完成!")
    print("="*60)
    print(f"成功: {results['success']}")
    print(f"失败: {results['failed']}")
    
    if results['details']:
        feat_diffs = [d['feature_diff'] for d in results['details']]
        cosine_sims = [d['cosine_sim'] for d in results['details']]
        
        print(f"\n特征变化统计:")
        print(f"  平均L2距离: {np.mean(feat_diffs):.4f}")
        print(f"  平均余弦相似度: {np.mean(cosine_sims):.4f}")
        print(f"  最大L2距离: {np.max(feat_diffs):.4f}")
        print(f"  最小余弦相似度: {np.min(cosine_sims):.4f}")
    
    # 保存结果
    result_path = output_dir / 'attack_results.json'
    with open(result_path, 'w') as f:
        json.dump({
            'method': 'transfer',
            'attack_type': args.attack_type,
            'epsilon': args.epsilon,
            'max_iter': args.max_iter,
            'step_size': args.step_size,
            'results': results
        }, f, indent=2)
    
    print(f"\n结果已保存到: {result_path}")
    
    # 打印下一步指令
    print("\n" + "="*60)
    print("下一步操作:")
    print("="*60)
    print(f"1. 提取攻击后图片的CLIP特征:")
    print(f"   python evaluate_attack.py --mode extract --split {args.split}")
    print(f"\n2. 在VIP5模型上评估攻击效果:")
    print(f"   python evaluate_attack_vip5.py --split {args.split} --num_samples 500")


if __name__ == '__main__':
    main()