#!/usr/bin/env python3
"""
VIP5 黑盒对抗攻击 - Transfer Attack (Fixed Version)
====================================================
基于CLIP的转移攻击方法 - 修复版

修复内容：
1. 正确加载目标特征（使用特征文件名而不是数字ID）
2. 改进攻击策略：向热门商品特征方向移动
3. 增加攻击强度和迭代次数

使用方法：
    python transfer_attack.py --split toys --num_images 100
    python transfer_attack.py --split toys --num_images 100 --epsilon 0.1 --max_iter 100

作者: 对抗攻击研究
"""

import os
import sys
import argparse
import pickle
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random
from typing import List, Dict, Optional
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import clip

SCRIPT_DIR = Path(__file__).resolve().parent


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def pil_to_tensor(img: Image.Image, device='cuda', size=224) -> torch.Tensor:
    """将PIL图像转换为tensor (0-1范围)"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
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
    基于CLIP的转移攻击器 - 修复版

    攻击策略：
    1. toward_popular: 向热门商品特征方向移动（推荐）
    2. maximize_similarity: 最大化与多个热门商品的相似度
    3. feature_enhancement: 增强特征显著性
    """

    def __init__(self, device='cuda', epsilon=0.1, max_iter=100,
                 step_size=None, momentum=0.9, attack_type='toward_popular'):
        """
        初始化攻击器

        Args:
            device: 计算设备
            epsilon: 最大扰动幅度 (建议0.08-0.15)
            max_iter: 迭代次数 (建议50-150)
            step_size: 步长 (默认: 2*epsilon/max_iter)
            momentum: 动量系数 (建议0.9)
            attack_type: 攻击类型
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.step_size = step_size if step_size else 2.0 * epsilon / max_iter
        self.momentum = momentum
        self.attack_type = attack_type

        # 目标特征
        self.target_features = None  # 平均目标特征
        self.target_feature_list = []  # 多个目标特征

        # 加载CLIP模型
        print(f"Loading CLIP ViT-B/32 on {self.device}...")
        self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device=self.device)
        self.clip_model.eval()
        print("Ready.")

    def extract_feature(self, image: Image.Image) -> np.ndarray:
        """提取CLIP特征"""
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_image(image_input)
        return features.cpu().numpy().squeeze()

    def set_target_features(self, avg_feature: np.ndarray, feature_list: List[np.ndarray] = None):
        """设置目标特征"""
        self.target_features = torch.from_numpy(avg_feature).float().to(self.device)
        if feature_list:
            self.target_feature_list = [
                torch.from_numpy(f).float().to(self.device) for f in feature_list
            ]
        print(f"Target features set: avg shape {avg_feature.shape}, {len(self.target_feature_list)} individual targets")

    def attack_single(self, image: Image.Image) -> Image.Image:
        """
        对单张图像执行攻击

        使用动量迭代FGSM (MI-FGSM) 向目标特征方向移动
        """
        # 转换为tensor
        img_tensor = pil_to_tensor(image, self.device)
        original_tensor = img_tensor.clone()

        # 获取原始特征
        with torch.no_grad():
            original_normalized = normalize_for_clip(img_tensor)
            original_feat = self.clip_model.encode_image(original_normalized)

        # 初始化
        perturbed = img_tensor.clone()
        momentum_grad = torch.zeros_like(perturbed)

        # MI-FGSM迭代
        for i in range(self.max_iter):
            perturbed.requires_grad = True

            # 前向传播
            normalized = normalize_for_clip(perturbed)
            features = self.clip_model.encode_image(normalized)
            features_norm = F.normalize(features, p=2, dim=1)

            # 计算损失
            if self.attack_type == 'toward_popular' and self.target_features is not None:
                # 向平均目标特征靠近
                target_norm = F.normalize(self.target_features.unsqueeze(0), p=2, dim=1)
                loss = 1.0 - F.cosine_similarity(features_norm, target_norm, dim=1).mean()

            elif self.attack_type == 'maximize_similarity' and len(self.target_feature_list) > 0:
                # 最大化与多个热门商品的相似度
                total_sim = 0.0
                for target_feat in self.target_feature_list:
                    target_norm = F.normalize(target_feat.unsqueeze(0), p=2, dim=1)
                    total_sim += F.cosine_similarity(features_norm, target_norm, dim=1).mean()
                loss = -total_sim / len(self.target_feature_list)

            elif self.attack_type == 'feature_enhancement':
                # 增强特征显著性（增加特征范数）+ 向目标靠近
                norm_loss = -torch.norm(features, p=2, dim=1).mean() * 0.1
                if self.target_features is not None:
                    target_norm = F.normalize(self.target_features.unsqueeze(0), p=2, dim=1)
                    sim_loss = 1.0 - F.cosine_similarity(features_norm, target_norm, dim=1).mean()
                    loss = sim_loss + norm_loss
                else:
                    loss = norm_loss

            else:
                # 默认：远离原始特征（不推荐，效果不稳定）
                orig_norm = F.normalize(original_feat, p=2, dim=1)
                loss = F.cosine_similarity(features_norm, orig_norm, dim=1).mean()

            # 反向传播
            loss.backward()

            with torch.no_grad():
                # 获取梯度
                grad = perturbed.grad

                # 归一化梯度 (L1范数)
                grad_norm = grad / (torch.abs(grad).mean(dim=[1, 2, 3], keepdim=True) + 1e-8)

                # 更新动量
                momentum_grad = self.momentum * momentum_grad + grad_norm

                # 使用动量梯度的符号进行更新
                perturbed = perturbed - self.step_size * momentum_grad.sign()

                # 投影到epsilon球内
                delta = perturbed - original_tensor
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)

                # 确保像素值有效
                perturbed = torch.clamp(original_tensor + delta, 0, 1)

            perturbed = perturbed.detach()

        # 转换回PIL并恢复原始大小
        attacked_img = tensor_to_pil(perturbed)
        attacked_img = attacked_img.resize(image.size, Image.LANCZOS)

        return attacked_img

    def attack_batch(self, image_dir: Path, output_dir: Path,
                     num_images: Optional[int] = None) -> Dict:
        """批量攻击图像"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有图像文件
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))

        # 随机采样
        if num_images and num_images < len(image_files):
            random.seed(42)
            image_files = random.sample(image_files, num_images)

        print(f"\n{'='*60}")
        print(f"Attack: {len(image_files)} images")
        print(f"{'='*60}")
        print(f"Attack Type: {self.attack_type}")
        print(f"Epsilon: {self.epsilon}")
        print(f"Max Iterations: {self.max_iter}")
        print(f"Step Size: {self.step_size:.6f}")
        print(f"Momentum: {self.momentum}")
        print(f"Target features: {'Set' if self.target_features is not None else 'Not set'}")
        print(f"{'='*60}\n")

        results = {
            'success': 0,
            'failed': 0,
            'feature_changes': [],
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

                # 计算与目标的相似度变化
                if self.target_features is not None:
                    target_np = self.target_features.cpu().numpy()
                    orig_target_sim = np.dot(original_feat, target_np) / (
                        np.linalg.norm(original_feat) * np.linalg.norm(target_np) + 1e-8
                    )
                    attack_target_sim = np.dot(attacked_feat, target_np) / (
                        np.linalg.norm(attacked_feat) * np.linalg.norm(target_np) + 1e-8
                    )
                    target_sim_change = attack_target_sim - orig_target_sim
                else:
                    target_sim_change = 0.0

                results['success'] += 1
                results['feature_changes'].append(feat_diff)
                results['details'].append({
                    'image': img_path.name,
                    'feature_diff': float(feat_diff),
                    'cosine_sim': float(cosine_sim),
                    'target_sim_change': float(target_sim_change)
                })

            except Exception as e:
                print(f"\nError processing {img_path.name}: {e}")
                results['failed'] += 1

        return results


def load_target_features_from_dir(feature_dir: Path, num_targets: int = 50) -> tuple:
    """
    直接从特征目录加载目标特征

    策略：加载所有特征，计算平均特征作为目标
    这样可以让攻击后的图像特征向"平均"方向移动，
    可能增加与其他商品的相似度
    """
    if not feature_dir.exists():
        print(f"Feature directory not found: {feature_dir}")
        return None, []

    feature_files = list(feature_dir.glob('*.npy'))
    print(f"Found {len(feature_files)} feature files in {feature_dir}")

    if len(feature_files) == 0:
        return None, []

    # 随机选择一部分作为目标
    random.seed(42)
    if len(feature_files) > num_targets:
        selected_files = random.sample(feature_files, num_targets)
    else:
        selected_files = feature_files

    features = []
    for f in selected_files:
        try:
            feat = np.load(f)
            features.append(feat)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if len(features) == 0:
        return None, []

    # 计算平均特征
    avg_feature = np.mean(features, axis=0)

    # 取前几个作为多目标
    top_features = features[:min(10, len(features))]

    print(f"Loaded {len(features)} target features, avg shape: {avg_feature.shape}")
    return avg_feature, top_features


def get_popular_item_features(split: str, feature_dir: Path, top_k: int = 30) -> tuple:
    """
    获取热门商品的特征

    步骤：
    1. 从sequential_data.txt统计商品出现频率
    2. 建立 商品ID -> 特征文件名 的映射
    3. 加载热门商品的特征
    """
    data_dir = SCRIPT_DIR / 'data' / split

    # 统计商品频率
    sequential_path = data_dir / 'sequential_data.txt'
    if not sequential_path.exists():
        print(f"Sequential data not found: {sequential_path}")
        return None, []

    item_counts = Counter()
    with open(sequential_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) > 1:
                items = parts[1:]
                for item in items:
                    item_counts[item] += 1

    popular_items = [item for item, _ in item_counts.most_common(top_k * 2)]
    print(f"Found {len(popular_items)} popular items")

    # 建立映射: 数字ID -> 图片文件名
    item2img_path = data_dir / 'item2img_dict.pkl'
    datamaps_path = data_dir / 'datamaps.json'

    if not item2img_path.exists() or not datamaps_path.exists():
        print("Mapping files not found, using direct feature loading")
        return load_target_features_from_dir(feature_dir, top_k)

    item2img_dict = load_pickle(str(item2img_path))
    datamaps = load_json(str(datamaps_path))
    id2item = datamaps.get('id2item', {})

    # 建立 数字ID -> 图片文件名 的映射
    id_to_imgname = {}
    for asin_key, img_info in item2img_dict.items():
        # 获取图片文件名
        if isinstance(img_info, str):
            img_filename = img_info
        elif isinstance(img_info, list) and len(img_info) > 0:
            img_filename = img_info[0]
        else:
            continue

        # 清理文件名
        if '/' in img_filename:
            img_filename = img_filename.split('/')[-1]
        if '.' in img_filename:
            img_filename = img_filename.rsplit('.', 1)[0]

        # 找到对应的数字ID
        for num_id, asin in id2item.items():
            if asin == asin_key:
                id_to_imgname[num_id] = img_filename
                break

    print(f"Built ID to image name mapping: {len(id_to_imgname)} entries")

    # 加载热门商品特征
    features = []
    loaded_count = 0

    for item_id in popular_items:
        if loaded_count >= top_k:
            break

        img_name = id_to_imgname.get(str(item_id))
        if img_name is None:
            continue

        feat_path = feature_dir / f"{img_name}.npy"
        if feat_path.exists():
            try:
                feat = np.load(feat_path)
                features.append(feat)
                loaded_count += 1
            except Exception as e:
                pass

    print(f"Loaded {len(features)} popular item features")

    if len(features) == 0:
        print("No popular item features loaded, using direct feature loading")
        return load_target_features_from_dir(feature_dir, top_k)

    avg_feature = np.mean(features, axis=0)
    top_features = features[:min(10, len(features))]

    return avg_feature, top_features


def main():
    parser = argparse.ArgumentParser(
        description='VIP5 黑盒对抗攻击 - Transfer Attack (Fixed)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法（向热门商品特征靠近）
  python transfer_attack.py --split toys --num_images 100

  # 更强的扰动
  python transfer_attack.py --split toys --num_images 100 --epsilon 0.12 --max_iter 150

  # 最大化与多个热门商品的相似度
  python transfer_attack.py --split toys --num_images 100 --attack_type maximize_similarity

参数建议:
  - epsilon: 0.08-0.15 (黑盒攻击需要较大扰动)
  - max_iter: 50-150 (更多迭代效果更好)
  - momentum: 0.9 (标准值)
        """
    )

    parser.add_argument('--split', type=str, default='toys',
                        help='数据集名称 (默认: toys)')
    parser.add_argument('--num_images', type=int, default=None,
                        help='攻击图像数量 (默认: 全部)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='最大扰动幅度 (默认: 0.1)')
    parser.add_argument('--max_iter', type=int, default=100,
                        help='迭代次数 (默认: 100)')
    parser.add_argument('--step_size', type=float, default=None,
                        help='步长 (默认: 2*epsilon/max_iter)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='动量系数 (默认: 0.9)')
    parser.add_argument('--attack_type', type=str, default='toward_popular',
                        choices=['toward_popular', 'maximize_similarity', 'feature_enhancement'],
                        help='攻击类型 (默认: toward_popular)')
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
    print("VIP5 Black-box Attack - Transfer Attack (Fixed)")
    print("="*60)
    print(f"Input: {image_dir}")
    print(f"Output: {output_dir}")
    print(f"Attack Type: {args.attack_type}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Max Iterations: {args.max_iter}")
    print(f"Momentum: {args.momentum}")

    # 创建攻击器
    attacker = TransferAttacker(
        device=args.device,
        epsilon=args.epsilon,
        max_iter=args.max_iter,
        step_size=args.step_size,
        momentum=args.momentum,
        attack_type=args.attack_type
    )

    # 加载目标特征
    avg_feat, feat_list = get_popular_item_features(args.split, feature_dir)
    if avg_feat is None:
        avg_feat, feat_list = load_target_features_from_dir(feature_dir)

    if avg_feat is not None:
        attacker.set_target_features(avg_feat, feat_list)
    else:
        print("\nWarning: No target features loaded!")
        print("Attack will use feature enhancement mode")
        attacker.attack_type = 'feature_enhancement'

    # 执行攻击
    results = attacker.attack_batch(image_dir, output_dir, args.num_images)

    # 打印结果
    print("\n" + "="*60)
    print("Attack Complete!")
    print("="*60)
    print(f"Success: {results['success']}")
    print(f"Failed: {results['failed']}")

    if results['details']:
        feat_diffs = [d['feature_diff'] for d in results['details']]
        cosine_sims = [d['cosine_sim'] for d in results['details']]
        target_changes = [d['target_sim_change'] for d in results['details']]

        print(f"\nFeature Change Statistics:")
        print(f"  Avg L2 Distance: {np.mean(feat_diffs):.4f}")
        print(f"  Avg Cosine Similarity: {np.mean(cosine_sims):.4f}")
        print(f"  Avg Target Sim Change: {np.mean(target_changes):+.4f}")
        print(f"  Positive Target Changes: {sum(1 for x in target_changes if x > 0)}/{len(target_changes)}")

    # 保存结果
    result_path = output_dir / 'attack_results.json'
    with open(result_path, 'w') as f:
        json.dump({
            'method': 'transfer_attack_fixed',
            'attack_type': args.attack_type,
            'epsilon': args.epsilon,
            'max_iter': args.max_iter,
            'momentum': args.momentum,
            'results': {
                'success': results['success'],
                'failed': results['failed'],
                'avg_feature_diff': float(np.mean(feat_diffs)) if feat_diffs else 0,
                'avg_target_sim_change': float(np.mean(target_changes)) if target_changes else 0,
            }
        }, f, indent=2)

    print(f"\nResults saved to: {result_path}")

    # 打印下一步指令
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print(f"1. Extract CLIP features:")
    print(f"   python evaluate_attack.py --mode extract --split {args.split}")
    print(f"\n2. Evaluate on VIP5:")
    print(f"   python evaluate_attack_vip5.py --split {args.split} --num_samples 500")


if __name__ == '__main__':
    main()
