#!/usr/bin/env python3
"""
VIP5 Black-box Transfer Attack v3
===================================
针对VIP5推荐系统的黑盒对抗攻击

核心设计原则：
1. 只用ViT-B/32 - VIP5只用这个模型，多模型集成产生冲突梯度反而有害
2. 匹配CLIP预处理 - 攻击时的预处理必须和evaluate_attack.py完全一致
3. 逐商品最近邻目标 - 每个商品攻击向最近的热门商品特征靠拢，不用平均
4. 直接L2+余弦组合损失 - 在原始特征空间优化，匹配VIP5的MLP输入
5. 保守扰动 - 小epsilon保持特征在训练分布内

使用方法：
    python transfer_attack.py --split toys
    python transfer_attack.py --split toys --num_images 100 --epsilon 0.06

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
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from collections import Counter

import clip

SCRIPT_DIR = Path(__file__).resolve().parent


def build_clip_preprocess_differentiable(n_px=224, device='cuda'):
    """
    构建与CLIP官方preprocess完全一致的可微分预处理

    CLIP preprocess = Resize(224, BICUBIC) + CenterCrop(224) + ToTensor() + Normalize
    对于已经是224x224的tensor，Resize+CenterCrop是恒等变换
    所以只需要保证输入是224x224然后做normalize
    """
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    def preprocess(tensor):
        return (tensor - mean) / std

    return preprocess


def pil_to_tensor_clip_style(img: Image.Image, n_px=224) -> torch.Tensor:
    """
    用CLIP的方式将PIL图像转换为tensor

    关键：先Resize最短边到n_px，再CenterCrop到n_px x n_px
    这和CLIP的preprocess完全一致
    """
    # Resize: 最短边缩放到n_px, 保持宽高比 (和CLIP一致)
    w, h = img.size
    if w < h:
        new_w = n_px
        new_h = int(h * n_px / w)
    else:
        new_h = n_px
        new_w = int(w * n_px / h)
    img_resized = img.resize((new_w, new_h), Image.BICUBIC)

    # CenterCrop到n_px x n_px
    left = (new_w - n_px) // 2
    top = (new_h - n_px) // 2
    img_cropped = img_resized.crop((left, top, left + n_px, top + n_px))

    # ToTensor
    tensor = transforms.ToTensor()(img_cropped)
    return tensor.unsqueeze(0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """将tensor转换回PIL图像"""
    tensor = tensor.squeeze(0).cpu()
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)


class TransferAttacker:
    """
    V3 转移攻击器

    关键改进：
    1. 只用ViT-B/32，不用集成（避免冲突梯度）
    2. 逐商品选择最近邻热门商品作为目标（不用平均特征）
    3. 匹配CLIP的预处理流程
    4. L2 + 余弦组合损失
    5. 标准MI-FGSM，不加TI和DI（不需要跨模型迁移）
    """

    def __init__(self, device='cuda', epsilon=0.06, max_iter=300,
                 step_size=None, momentum=0.9):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.step_size = step_size if step_size else 1.5 * epsilon / max_iter
        self.momentum = momentum

        # 热门商品特征库（逐商品查找最近邻用）
        self.popular_features = None  # list of np.ndarray
        self.popular_features_tensor = None  # stacked tensor on device

        # 加载CLIP模型（只用ViT-B/32）
        self._load_model()

        # 可微分预处理
        self.diff_preprocess = build_clip_preprocess_differentiable(device=self.device)

    def _load_model(self):
        """只加载ViT-B/32"""
        print(f"Loading CLIP ViT-B/32 on {self.device}...")
        self.model, self.preprocess = clip.load('ViT-B/32', device=self.device)
        self.model.eval()
        print("CLIP model ready.")

    def extract_feature(self, image: Image.Image) -> np.ndarray:
        """使用CLIP官方preprocess提取特征（和evaluate_attack.py一致）"""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_input)
        return features.cpu().numpy().squeeze().astype(np.float32)

    def set_popular_features(self, features: List[np.ndarray]):
        """设置热门商品特征库"""
        self.popular_features = features
        stacked = np.stack(features, axis=0)
        self.popular_features_tensor = torch.from_numpy(stacked).float().to(self.device)
        print(f"Loaded {len(features)} popular item features as target pool")

    def _find_nearest_target(self, original_feat: np.ndarray) -> torch.Tensor:
        """
        为当前商品找到最近的热门商品特征作为攻击目标

        使用余弦相似度找最近邻（排除自身），然后取top-3的平均
        这样目标是"真实"的，VIP5的MLP见过类似的特征
        """
        if self.popular_features_tensor is None:
            return None

        feat_tensor = torch.from_numpy(original_feat).float().to(self.device).unsqueeze(0)

        # 计算与所有热门商品的余弦相似度
        feat_norm = F.normalize(feat_tensor, p=2, dim=1)
        pop_norm = F.normalize(self.popular_features_tensor, p=2, dim=1)
        similarities = torch.mm(feat_norm, pop_norm.t()).squeeze(0)

        # 排除自身（相似度>0.99的）
        mask = similarities < 0.99
        similarities_masked = similarities.clone()
        similarities_masked[~mask] = -1.0

        # 取top-3最近邻的加权平均
        top_k = min(3, mask.sum().item())
        if top_k == 0:
            # 全部被排除，取最近的一个
            idx = similarities.argmax()
            return self.popular_features_tensor[idx]

        _, top_indices = similarities_masked.topk(top_k)
        top_feats = self.popular_features_tensor[top_indices]
        top_sims = similarities_masked[top_indices]

        # 相似度加权平均 - 更近的目标权重更大
        weights = F.softmax(top_sims * 5.0, dim=0).unsqueeze(1)  # temperature=5
        target = (top_feats * weights).sum(dim=0)

        return target

    def _compute_loss(self, perturbed: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
        """
        计算攻击损失

        组合L2距离和余弦相似度：
        - L2：直接拉近原始空间距离（匹配VIP5 MLP输入）
        - 余弦：拉近方向（特征方向是推荐的关键信号）
        """
        # 提取当前特征
        normalized = self.diff_preprocess(perturbed)
        features = self.model.encode_image(normalized)

        target = target_feat.unsqueeze(0)

        # 余弦损失 (主要)
        feat_norm = F.normalize(features, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        cosine_loss = 1.0 - F.cosine_similarity(feat_norm, target_norm, dim=1).mean()

        # L2损失 (辅助 - 在原始空间拉近)
        l2_loss = F.mse_loss(features, target)

        # 组合
        loss = 0.7 * cosine_loss + 0.3 * l2_loss / (l2_loss.detach() + 1e-8)

        return loss

    def attack_single(self, image: Image.Image, target_feat: torch.Tensor) -> Image.Image:
        """
        对单张图像执行MI-FGSM攻击

        简单直接的MI-FGSM：
        - 只用ViT-B/32
        - 匹配CLIP预处理
        - 向最近邻热门商品特征靠拢
        """
        # 用CLIP方式转换为tensor
        img_tensor = pil_to_tensor_clip_style(image).to(self.device)
        original_tensor = img_tensor.clone()

        perturbed = img_tensor.clone()
        momentum_grad = torch.zeros_like(perturbed)

        for i in range(self.max_iter):
            perturbed.requires_grad = True

            loss = self._compute_loss(perturbed, target_feat)
            loss.backward()

            with torch.no_grad():
                grad = perturbed.grad

                # 归一化梯度
                grad_norm = grad / (torch.abs(grad).mean(dim=[1, 2, 3], keepdim=True) + 1e-8)

                # 更新动量
                momentum_grad = self.momentum * momentum_grad + grad_norm

                # 符号更新
                perturbed = perturbed - self.step_size * momentum_grad.sign()

                # 投影到epsilon球内
                delta = torch.clamp(perturbed - original_tensor, -self.epsilon, self.epsilon)
                perturbed = torch.clamp(original_tensor + delta, 0, 1)

            perturbed = perturbed.detach()

        # 转换回PIL
        attacked_img = tensor_to_pil(perturbed)

        # 恢复到原始大小（evaluate_attack.py会重新做CLIP preprocess）
        attacked_img = attacked_img.resize(image.size, Image.LANCZOS)

        return attacked_img

    def attack_batch(self, image_dir: Path, output_dir: Path,
                     num_images: Optional[int] = None) -> Dict:
        """批量攻击"""
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))

        if num_images and num_images < len(image_files):
            random.seed(42)
            image_files = random.sample(image_files, num_images)

        print(f"\n{'='*60}")
        print(f"V3 Attack: {len(image_files)} images")
        print(f"{'='*60}")
        print(f"  Model: ViT-B/32 only (matching VIP5)")
        print(f"  Epsilon: {self.epsilon}")
        print(f"  Iterations: {self.max_iter}")
        print(f"  Step Size: {self.step_size:.6f}")
        print(f"  Momentum: {self.momentum}")
        print(f"  Target: per-item nearest neighbor")
        print(f"  Popular pool: {len(self.popular_features) if self.popular_features else 0} items")
        print(f"{'='*60}\n")

        results = {'success': 0, 'failed': 0, 'details': []}

        for img_path in tqdm(image_files, desc="Attacking"):
            try:
                original_img = Image.open(img_path).convert('RGB')
                original_feat = self.extract_feature(original_img)

                # 为当前商品找最近邻目标
                target_feat = self._find_nearest_target(original_feat)
                if target_feat is None:
                    # 无目标，跳过
                    output_path = output_dir / img_path.name
                    original_img.save(output_path, quality=95)
                    results['details'].append({
                        'image': img_path.name,
                        'skipped': True,
                        'reason': 'no target features'
                    })
                    continue

                # 执行攻击
                attacked_img = self.attack_single(original_img, target_feat)
                attacked_feat = self.extract_feature(attacked_img)

                # 保存
                output_path = output_dir / img_path.name
                attacked_img.save(output_path, quality=95)

                # 统计
                feat_diff = np.linalg.norm(attacked_feat - original_feat)
                orig_cosine = np.dot(original_feat, attacked_feat) / (
                    np.linalg.norm(original_feat) * np.linalg.norm(attacked_feat) + 1e-8
                )
                target_np = target_feat.cpu().numpy()
                target_sim_before = np.dot(original_feat, target_np) / (
                    np.linalg.norm(original_feat) * np.linalg.norm(target_np) + 1e-8
                )
                target_sim_after = np.dot(attacked_feat, target_np) / (
                    np.linalg.norm(attacked_feat) * np.linalg.norm(target_np) + 1e-8
                )

                results['success'] += 1
                results['details'].append({
                    'image': img_path.name,
                    'feature_diff': float(feat_diff),
                    'cosine_with_original': float(orig_cosine),
                    'target_sim_before': float(target_sim_before),
                    'target_sim_after': float(target_sim_after),
                    'target_sim_gain': float(target_sim_after - target_sim_before),
                })

            except Exception as e:
                print(f"\nError: {img_path.name}: {e}")
                results['failed'] += 1

        return results


def load_popular_features(feature_dir: Path, split: str,
                          top_k: int = 50) -> List[np.ndarray]:
    """
    加载热门商品特征

    综合末尾频率和全局频率选择热门商品
    """
    data_dir = SCRIPT_DIR / 'data' / split
    sequential_path = data_dir / 'sequential_data.txt'

    if not sequential_path.exists():
        print(f"Warning: {sequential_path} not found")
        return []

    end_counts = Counter()
    global_counts = Counter()

    with open(sequential_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) > 1:
                items = parts[1:]
                for item in items:
                    global_counts[item] += 1
                for item in parts[-3:] if len(parts) > 3 else parts[1:]:
                    end_counts[item] += 1

    # 综合评分
    scores = {}
    for item in set(list(end_counts.keys()) + list(global_counts.keys())):
        scores[item] = end_counts.get(item, 0) * 2 + global_counts.get(item, 0)

    top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    features = []
    loaded = 0
    for item_id, score in top_items:
        feat_path = feature_dir / f"{item_id}.npy"
        if feat_path.exists():
            features.append(np.load(feat_path).astype(np.float32))
            loaded += 1
            if loaded >= top_k:
                break

    print(f"Loaded {len(features)} popular item features from {feature_dir}")
    return features


def main():
    parser = argparse.ArgumentParser(
        description='VIP5 Black-box Transfer Attack v3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用方法:
  # 默认参数（推荐）
  python transfer_attack.py --split toys

  # 指定数量
  python transfer_attack.py --split toys --num_images 100

  # 调整扰动大小
  python transfer_attack.py --split toys --epsilon 0.08

攻击完成后:
  1. python evaluate_attack.py --mode extract --split toys
  2. python evaluate_attack_vip5.py --split toys --num_samples 500
        """
    )

    parser.add_argument('--split', type=str, default='toys')
    parser.add_argument('--num_images', type=int, default=None)
    parser.add_argument('--epsilon', type=float, default=0.06,
                        help='扰动幅度 (默认0.06，建议0.04-0.08)')
    parser.add_argument('--max_iter', type=int, default=300,
                        help='迭代次数 (默认300)')
    parser.add_argument('--step_size', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--top_k_popular', type=int, default=50,
                        help='热门商品池大小 (默认50)')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    image_dir = SCRIPT_DIR / args.split
    output_dir = SCRIPT_DIR / f'{args.split}2'
    feature_dir = SCRIPT_DIR / 'features' / 'vitb32_features' / f'{args.split}_original'

    if not image_dir.exists():
        print(f"Error: {image_dir} not found")
        sys.exit(1)

    print("\n" + "="*60)
    print("VIP5 Black-box Transfer Attack v3")
    print("="*60)

    # 创建攻击器
    attacker = TransferAttacker(
        device=args.device,
        epsilon=args.epsilon,
        max_iter=args.max_iter,
        step_size=args.step_size,
        momentum=args.momentum,
    )

    # 加载热门商品特征作为目标池
    if feature_dir.exists():
        popular_feats = load_popular_features(
            feature_dir, args.split, top_k=args.top_k_popular
        )
        if popular_feats:
            attacker.set_popular_features(popular_feats)
    else:
        print(f"Warning: feature dir {feature_dir} not found.")
        print("Running without target features (attack will be less effective)")

    # 执行攻击
    results = attacker.attack_batch(image_dir, output_dir, args.num_images)

    # 统计
    print("\n" + "="*60)
    print("Attack Complete!")
    print("="*60)
    print(f"Success: {results['success']}, Failed: {results['failed']}")

    valid = [d for d in results['details'] if 'feature_diff' in d]
    if valid:
        feat_diffs = [d['feature_diff'] for d in valid]
        cosines = [d['cosine_with_original'] for d in valid]
        gains = [d['target_sim_gain'] for d in valid]
        positive_gains = [g for g in gains if g > 0]

        print(f"\nFeature Statistics:")
        print(f"  Avg L2 distance: {np.mean(feat_diffs):.4f}")
        print(f"  Avg cosine w/ original: {np.mean(cosines):.4f}")
        print(f"  Avg target sim gain: {np.mean(gains):+.4f}")
        print(f"  Items moved closer to target: {len(positive_gains)}/{len(gains)} ({100*len(positive_gains)/len(gains):.1f}%)")

    # 保存
    result_path = output_dir / 'attack_results.json'
    with open(result_path, 'w') as f:
        json.dump({
            'method': 'v3_nearest_neighbor',
            'epsilon': args.epsilon,
            'max_iter': args.max_iter,
            'momentum': args.momentum,
            'top_k_popular': args.top_k_popular,
            'results': results
        }, f, indent=2)
    print(f"\nResults saved to: {result_path}")

    print(f"\nNext steps:")
    print(f"  python evaluate_attack.py --mode extract --split {args.split}")
    print(f"  python evaluate_attack_vip5.py --split {args.split} --num_samples 500")


if __name__ == '__main__':
    main()
