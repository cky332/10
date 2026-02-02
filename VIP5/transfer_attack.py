#!/usr/bin/env python3
"""
VIP5 Enhanced Black-box Attack
==============================
增强版黑盒对抗攻击 - 提高转移攻击成功率

改进点：
1. 多模型集成攻击 (Ensemble Attack) - 同时攻击多个CLIP变体
2. 动量迭代FGSM (MI-FGSM) - 稳定梯度方向,提高转移性
3. 输入多样性攻击 (DI-FGSM) - 随机变换增强泛化
4. 频率域扰动 (Spectral Attack) - 针对低频特征
5. 改进的目标特征选择 - 基于特征分布分析

使用方法：
    python enhanced_attack.py --split toys --num_images 100
    python enhanced_attack.py --split toys --num_images 100 --epsilon 0.1 --max_iter 100

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
from collections import Counter

# 导入CLIP
import clip

SCRIPT_DIR = Path(__file__).resolve().parent


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


class InputDiversityTransform:
    """输入多样性变换 - 增强转移攻击效果"""

    def __init__(self, prob=0.7, scale_range=(0.9, 1.1)):
        self.prob = prob
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.prob:
            return x

        # 随机缩放
        scale = random.uniform(*self.scale_range)
        new_size = int(224 * scale)

        # 缩放后padding回224
        x_resized = F.interpolate(x, size=(new_size, new_size), mode='bilinear', align_corners=False)

        if new_size < 224:
            # 需要padding
            pad_total = 224 - new_size
            pad_left = random.randint(0, pad_total)
            pad_top = random.randint(0, pad_total)
            pad_right = pad_total - pad_left
            pad_bottom = pad_total - pad_top
            x_padded = F.pad(x_resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        else:
            # 需要裁剪
            crop_total = new_size - 224
            crop_left = random.randint(0, crop_total)
            crop_top = random.randint(0, crop_total)
            x_padded = x_resized[:, :, crop_top:crop_top+224, crop_left:crop_left+224]

        return x_padded


class EnhancedTransferAttacker:
    """
    增强版转移攻击器

    核心改进：
    1. 多模型集成：同时攻击多个CLIP模型,提高转移性
    2. 动量攻击 (MI-FGSM)：累积梯度动量,稳定优化方向
    3. 输入多样性 (DI-FGSM)：随机变换输入,增强泛化
    4. 改进目标函数：基于特征分布设计更有效的目标
    """

    def __init__(self, device='cuda', epsilon=0.1, max_iter=100,
                 step_size=None, momentum=0.9, use_ensemble=True,
                 use_input_diversity=True, attack_type='enhanced'):
        """
        初始化攻击器

        Args:
            device: 计算设备
            epsilon: 最大扰动幅度 (建议0.08-0.15)
            max_iter: 迭代次数 (建议50-200)
            step_size: 步长 (默认: 2*epsilon/max_iter)
            momentum: 动量系数 (建议0.9)
            use_ensemble: 是否使用模型集成
            use_input_diversity: 是否使用输入多样性
            attack_type: 攻击类型
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.step_size = step_size if step_size else 2 * epsilon / max_iter
        self.momentum = momentum
        self.use_ensemble = use_ensemble
        self.use_input_diversity = use_input_diversity
        self.attack_type = attack_type

        # 输入多样性变换
        self.input_diversity = InputDiversityTransform(prob=0.7)

        # 目标特征
        self.target_features = None
        self.feature_directions = None

        # 加载CLIP模型
        self._load_models()

    def _load_models(self):
        """加载CLIP模型集成"""
        print(f"Loading CLIP models on {self.device}...")

        self.models = []
        self.preprocessors = []
        self.model_weights = []

        # 主模型: ViT-B/32 (VIP5使用的模型)
        model, preprocess = clip.load('ViT-B/32', device=self.device)
        model.eval()
        self.models.append(model)
        self.preprocessors.append(preprocess)
        self.model_weights.append(1.0)  # 主模型权重最高

        if self.use_ensemble:
            # 尝试加载其他模型提高转移性
            try:
                model2, preprocess2 = clip.load('ViT-B/16', device=self.device)
                model2.eval()
                self.models.append(model2)
                self.preprocessors.append(preprocess2)
                self.model_weights.append(0.5)
                print("  Loaded ViT-B/16")
            except Exception as e:
                print(f"  Could not load ViT-B/16: {e}")

            try:
                model3, preprocess3 = clip.load('RN50', device=self.device)
                model3.eval()
                self.models.append(model3)
                self.preprocessors.append(preprocess3)
                self.model_weights.append(0.3)
                print("  Loaded RN50")
            except Exception as e:
                print(f"  Could not load RN50: {e}")

        # 归一化权重
        total_weight = sum(self.model_weights)
        self.model_weights = [w / total_weight for w in self.model_weights]

        print(f"Loaded {len(self.models)} CLIP models")

    def extract_feature(self, image: Image.Image, model_idx=0) -> np.ndarray:
        """使用指定模型提取CLIP特征"""
        model = self.models[model_idx]
        preprocess = self.preprocessors[model_idx]

        image_input = preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = model.encode_image(image_input)
        return features.cpu().numpy().squeeze()

    def set_target_features(self, features: np.ndarray):
        """设置目标特征"""
        self.target_features = torch.from_numpy(features).float().to(self.device)
        print(f"Target features set, shape: {features.shape}")

    def set_feature_directions(self, directions: List[np.ndarray]):
        """设置多个目标特征方向"""
        self.feature_directions = [
            torch.from_numpy(d).float().to(self.device) for d in directions
        ]
        print(f"Set {len(directions)} feature directions")

    def _compute_ensemble_loss(self, perturbed: torch.Tensor, original_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        计算集成损失

        组合多个攻击目标：
        1. 远离原始特征 (让图像特征偏离原始)
        2. 靠近目标特征 (如果设置了目标)
        3. 增加与多个方向的相似度 (如果设置了方向)
        """
        total_loss = 0.0

        for i, (model, weight) in enumerate(zip(self.models, self.model_weights)):
            # 应用输入多样性
            if self.use_input_diversity:
                perturbed_div = self.input_diversity(perturbed)
            else:
                perturbed_div = perturbed

            # 提取当前特征
            normalized = normalize_for_clip(perturbed_div)
            features = model.encode_image(normalized)
            features = F.normalize(features, p=2, dim=1)

            orig_feat = F.normalize(original_feats[i], p=2, dim=1)

            # 损失1: 远离原始特征
            loss_away = F.cosine_similarity(features, orig_feat, dim=1).mean()

            # 损失2: 靠近目标特征 (如果有)
            loss_toward = 0.0
            if self.target_features is not None:
                target = F.normalize(self.target_features.unsqueeze(0), p=2, dim=1)
                loss_toward = 1.0 - F.cosine_similarity(features, target, dim=1).mean()

            # 损失3: 靠近多个特征方向 (如果有)
            loss_directions = 0.0
            if self.feature_directions is not None:
                for direction in self.feature_directions:
                    dir_feat = F.normalize(direction.unsqueeze(0), p=2, dim=1)
                    loss_directions += 1.0 - F.cosine_similarity(features, dir_feat, dim=1).mean()
                loss_directions /= len(self.feature_directions)

            # 组合损失 (权重可调)
            if self.attack_type == 'toward_target' and self.target_features is not None:
                # 主要优化靠近目标
                model_loss = 0.3 * loss_away + 0.7 * loss_toward
            elif self.attack_type == 'multi_direction' and self.feature_directions is not None:
                # 优化多方向
                model_loss = 0.2 * loss_away + 0.8 * loss_directions
            else:
                # 默认: 远离原始 + 靠近目标 (如果有)
                if self.target_features is not None:
                    model_loss = 0.4 * loss_away + 0.6 * loss_toward
                else:
                    model_loss = loss_away

            total_loss += weight * model_loss

        return total_loss

    def attack_single(self, image: Image.Image) -> Image.Image:
        """
        对单张图像执行增强攻击

        使用 MI-FGSM (动量迭代FGSM) + 输入多样性 + 模型集成
        """
        # 转换为tensor
        img_tensor = pil_to_tensor(image, self.device)
        original_tensor = img_tensor.clone()

        # 获取各模型的原始特征
        original_feats = []
        for model in self.models:
            with torch.no_grad():
                normalized = normalize_for_clip(img_tensor)
                feat = model.encode_image(normalized)
                original_feats.append(feat.clone())

        # 初始化
        perturbed = img_tensor.clone()
        momentum_grad = torch.zeros_like(perturbed)

        # MI-FGSM 迭代
        for i in range(self.max_iter):
            perturbed.requires_grad = True

            # 计算集成损失
            loss = self._compute_ensemble_loss(perturbed, original_feats)

            # 反向传播
            loss.backward()

            with torch.no_grad():
                # 获取梯度
                grad = perturbed.grad

                # 归一化梯度 (L1范数)
                grad_norm = grad / (torch.abs(grad).mean(dim=[1,2,3], keepdim=True) + 1e-8)

                # 更新动量
                momentum_grad = self.momentum * momentum_grad + grad_norm

                # 使用动量梯度的符号进行更新
                grad_sign = momentum_grad.sign()
                perturbed = perturbed - self.step_size * grad_sign

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
        print(f"Enhanced Attack: Processing {len(image_files)} images")
        print(f"{'='*60}")
        print(f"Attack Type: {self.attack_type}")
        print(f"Epsilon: {self.epsilon}")
        print(f"Max Iterations: {self.max_iter}")
        print(f"Step Size: {self.step_size:.6f}")
        print(f"Momentum: {self.momentum}")
        print(f"Use Ensemble: {self.use_ensemble}")
        print(f"Use Input Diversity: {self.use_input_diversity}")
        print(f"Number of Models: {len(self.models)}")
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
    """获取热门商品列表"""
    data_dir = SCRIPT_DIR / 'data' / split
    sequential_path = data_dir / 'sequential_data.txt'

    if not sequential_path.exists():
        print(f"Warning: {sequential_path} not found")
        return []

    item_counts = Counter()

    with open(sequential_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) > 1:
                items = parts[1:]
                for item in items:
                    item_counts[item] += 1

    popular = [item for item, _ in item_counts.most_common(top_k)]
    print(f"Found {len(popular)} popular items")
    return popular


def load_target_features(feature_dir: Path, popular_items: List[str],
                         num_targets: int = 30) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
    """
    加载目标特征

    返回:
    - 平均目标特征
    - 多个目标特征方向(用于多方向攻击)
    """
    features = []

    for item_id in popular_items[:num_targets]:
        feat_path = feature_dir / f"{item_id}.npy"
        if feat_path.exists():
            features.append(np.load(feat_path))

    if len(features) == 0:
        print("Warning: No target features found")
        return None, []

    # 平均特征
    avg_feature = np.mean(features, axis=0)

    # 多个目标方向 (取最热门的几个)
    top_directions = features[:min(10, len(features))]

    print(f"Loaded {len(features)} target features, {len(top_directions)} top directions")
    return avg_feature, top_directions


def analyze_high_rank_features(feature_dir: Path, split: str) -> Optional[np.ndarray]:
    """
    分析高排名商品的特征分布，提取有效攻击方向

    这是一个更智能的目标特征选择策略
    """
    data_dir = SCRIPT_DIR / 'data' / split
    sequential_path = data_dir / 'sequential_data.txt'

    if not sequential_path.exists():
        return None

    # 统计商品出现在用户序列末尾的频率（这些可能是被推荐的商品）
    end_item_counts = Counter()

    with open(sequential_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) > 1:
                # 序列末尾的商品更可能是被推荐的
                end_items = parts[-3:] if len(parts) > 3 else parts[1:]
                for item in end_items:
                    end_item_counts[item] += 1

    # 获取频繁出现在末尾的商品
    top_end_items = [item for item, _ in end_item_counts.most_common(50)]

    # 加载这些商品的特征
    features = []
    for item_id in top_end_items:
        feat_path = feature_dir / f"{item_id}.npy"
        if feat_path.exists():
            features.append(np.load(feat_path))

    if len(features) < 5:
        return None

    # 计算这些"高排名"商品的平均特征
    avg_feature = np.mean(features, axis=0)

    print(f"Analyzed {len(features)} high-rank items for target feature")
    return avg_feature


def main():
    parser = argparse.ArgumentParser(
        description='VIP5 Enhanced Black-box Attack',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法 (增强攻击)
  python enhanced_attack.py --split toys --num_images 100

  # 更强的扰动
  python enhanced_attack.py --split toys --num_images 100 --epsilon 0.12 --max_iter 150

  # 向热门商品特征靠近
  python enhanced_attack.py --split toys --num_images 100 --attack_type toward_target

  # 多方向攻击
  python enhanced_attack.py --split toys --num_images 100 --attack_type multi_direction

  # 禁用集成/输入多样性 (更快但效果可能较差)
  python enhanced_attack.py --split toys --num_images 100 --no_ensemble --no_diversity

参数建议:
  - epsilon: 0.08-0.15 (黑盒攻击需要更大扰动)
  - max_iter: 50-200 (更多迭代提高效果)
  - momentum: 0.9 (MI-FGSM标准值)
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
    parser.add_argument('--attack_type', type=str, default='enhanced',
                        choices=['enhanced', 'toward_target', 'multi_direction'],
                        help='攻击类型 (默认: enhanced)')
    parser.add_argument('--no_ensemble', action='store_true',
                        help='禁用模型集成')
    parser.add_argument('--no_diversity', action='store_true',
                        help='禁用输入多样性')
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
    print("VIP5 Enhanced Black-box Attack")
    print("="*60)
    print(f"Input Directory: {image_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Attack Type: {args.attack_type}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Max Iterations: {args.max_iter}")
    print(f"Momentum: {args.momentum}")
    print(f"Use Ensemble: {not args.no_ensemble}")
    print(f"Use Input Diversity: {not args.no_diversity}")

    # 创建攻击器
    attacker = EnhancedTransferAttacker(
        device=args.device,
        epsilon=args.epsilon,
        max_iter=args.max_iter,
        step_size=args.step_size,
        momentum=args.momentum,
        use_ensemble=not args.no_ensemble,
        use_input_diversity=not args.no_diversity,
        attack_type=args.attack_type
    )

    # 加载目标特征
    if feature_dir.exists():
        # 首先尝试分析高排名商品特征
        smart_target = analyze_high_rank_features(feature_dir, args.split)

        if smart_target is not None:
            attacker.set_target_features(smart_target)
        else:
            # 回退到热门商品特征
            popular_items = get_popular_items(args.split)
            if popular_items:
                avg_feat, directions = load_target_features(feature_dir, popular_items)
                if avg_feat is not None:
                    attacker.set_target_features(avg_feat)
                if directions and args.attack_type == 'multi_direction':
                    attacker.set_feature_directions(directions)

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

        print(f"\nFeature Change Statistics:")
        print(f"  Avg L2 Distance: {np.mean(feat_diffs):.4f}")
        print(f"  Avg Cosine Similarity: {np.mean(cosine_sims):.4f}")
        print(f"  Max L2 Distance: {np.max(feat_diffs):.4f}")
        print(f"  Min Cosine Similarity: {np.min(cosine_sims):.4f}")

    # 保存结果
    result_path = output_dir / 'attack_results.json'
    with open(result_path, 'w') as f:
        json.dump({
            'method': 'enhanced_transfer',
            'attack_type': args.attack_type,
            'epsilon': args.epsilon,
            'max_iter': args.max_iter,
            'momentum': args.momentum,
            'use_ensemble': not args.no_ensemble,
            'use_input_diversity': not args.no_diversity,
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to: {result_path}")

    # 打印下一步指令
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print(f"1. Extract CLIP features from attacked images:")
    print(f"   python evaluate_attack.py --mode extract --split {args.split}")
    print(f"\n2. Evaluate attack effect on VIP5:")
    print(f"   python evaluate_attack_vip5.py --split {args.split} --num_samples 500")


if __name__ == '__main__':
    main()
