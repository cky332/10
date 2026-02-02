#!/usr/bin/env python3
"""
VIP5 Enhanced Black-box Attack (Optimized)
===========================================
优化版黑盒对抗攻击 - 大幅提高转移攻击成功率

核心优化点：
1. 纯目标导向损失 - 移除"远离原始"损失，避免随机方向扰动导致排名下降
2. TI-FGSM (Translation-Invariant) - 高斯核平滑梯度，提高转移性
3. 自适应目标特征选择 - 基于特征聚类选择更有效的目标方向
4. 增大扰动预算和迭代次数
5. 余弦相似度优化 - 直接最大化与目标的余弦相似度

使用方法：
    python transfer_attack.py --split toys --num_images 100
    python transfer_attack.py --split toys --num_images 100 --epsilon 0.12 --max_iter 200

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


def create_gaussian_kernel(kernel_size=5, sigma=1.0, channels=3, device='cuda'):
    """创建高斯核用于TI-FGSM梯度平滑"""
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return kernel


class InputDiversityTransform:
    """输入多样性变换 - 增强转移攻击效果"""

    def __init__(self, prob=0.7, scale_range=(0.85, 1.15)):
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
    优化版转移攻击器

    核心改进：
    1. 纯目标导向损失 - 只优化"靠近目标"，不再"远离原始"
    2. TI-FGSM - 高斯核平滑梯度，大幅提高转移性
    3. 多模型集成 + 输入多样性
    4. 自适应步长衰减
    5. 多目标特征加权融合
    """

    def __init__(self, device='cuda', epsilon=0.12, max_iter=200,
                 step_size=None, momentum=0.9, use_ensemble=True,
                 use_input_diversity=True, attack_type='toward_target',
                 ti_kernel_size=5, ti_sigma=1.0):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.step_size = step_size if step_size else 2.5 * epsilon / max_iter
        self.momentum = momentum
        self.use_ensemble = use_ensemble
        self.use_input_diversity = use_input_diversity
        self.attack_type = attack_type

        # TI-FGSM 高斯核参数
        self.ti_kernel_size = ti_kernel_size
        self.ti_sigma = ti_sigma

        # 输入多样性变换 - 更大范围
        self.input_diversity = InputDiversityTransform(prob=0.7, scale_range=(0.85, 1.15))

        # 目标特征
        self.target_features = None
        self.feature_directions = None
        self.target_features_per_model = None  # 每个模型的目标特征

        # 加载CLIP模型
        self._load_models()

        # 创建高斯核
        self.gaussian_kernel = create_gaussian_kernel(
            kernel_size=ti_kernel_size, sigma=ti_sigma,
            channels=3, device=self.device
        )

    def _load_models(self):
        """加载CLIP模型集成"""
        print(f"Loading CLIP models on {self.device}...")

        self.models = []
        self.preprocessors = []
        self.model_weights = []

        # 主模型: ViT-B/32 (VIP5使用的模型) - 最高权重
        model, preprocess = clip.load('ViT-B/32', device=self.device)
        model.eval()
        self.models.append(model)
        self.preprocessors.append(preprocess)
        self.model_weights.append(1.0)

        if self.use_ensemble:
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

    def set_target_features_per_model(self, features_list: List[np.ndarray]):
        """为每个模型设置对应的目标特征（更精确的集成攻击）"""
        self.target_features_per_model = [
            torch.from_numpy(f).float().to(self.device) for f in features_list
        ]
        print(f"Set per-model target features for {len(features_list)} models")

    def _smooth_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """TI-FGSM: 使用高斯核平滑梯度，提高转移性"""
        padding = self.ti_kernel_size // 2
        smoothed = F.conv2d(grad, self.gaussian_kernel, padding=padding, groups=3)
        return smoothed

    def _compute_toward_loss(self, perturbed: torch.Tensor, original_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        纯目标导向损失 - 只优化靠近目标特征

        关键改进：不再包含"远离原始"损失，避免随机方向扰动
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

            # 选择目标特征
            if self.target_features_per_model is not None and i < len(self.target_features_per_model):
                target = self.target_features_per_model[i]
            elif self.target_features is not None:
                target = self.target_features
            else:
                # 无目标时回退到远离原始（不推荐）
                orig_feat = F.normalize(original_feats[i], p=2, dim=1)
                total_loss += weight * F.cosine_similarity(features, orig_feat, dim=1).mean()
                continue

            target = F.normalize(target.unsqueeze(0), p=2, dim=1)

            # 核心损失：最大化与目标的余弦相似度
            loss_toward = 1.0 - F.cosine_similarity(features, target, dim=1).mean()

            # 如果有多个方向目标，额外添加方向损失
            loss_directions = 0.0
            if self.feature_directions is not None and self.attack_type == 'multi_direction':
                for direction in self.feature_directions:
                    dir_feat = F.normalize(direction.unsqueeze(0), p=2, dim=1)
                    loss_directions += 1.0 - F.cosine_similarity(features, dir_feat, dim=1).mean()
                loss_directions /= len(self.feature_directions)
                # 组合：主要目标 + 辅助方向
                model_loss = 0.6 * loss_toward + 0.4 * loss_directions
            else:
                model_loss = loss_toward

            total_loss += weight * model_loss

        return total_loss

    def attack_single(self, image: Image.Image) -> Image.Image:
        """
        对单张图像执行优化攻击

        使用 MI-FGSM + TI-FGSM + 输入多样性 + 模型集成
        纯目标导向损失
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

        # MI-FGSM + TI-FGSM 迭代
        for i in range(self.max_iter):
            perturbed.requires_grad = True

            # 计算纯目标导向损失
            loss = self._compute_toward_loss(perturbed, original_feats)

            # 反向传播
            loss.backward()

            with torch.no_grad():
                # 获取梯度
                grad = perturbed.grad

                # TI-FGSM: 高斯核平滑梯度
                grad = self._smooth_gradient(grad)

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
        print(f"Optimized Attack: Processing {len(image_files)} images")
        print(f"{'='*60}")
        print(f"Attack Type: {self.attack_type}")
        print(f"Epsilon: {self.epsilon}")
        print(f"Max Iterations: {self.max_iter}")
        print(f"Step Size: {self.step_size:.6f}")
        print(f"Momentum: {self.momentum}")
        print(f"TI-FGSM Kernel: {self.ti_kernel_size}x{self.ti_kernel_size}, sigma={self.ti_sigma}")
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

                # 如果有目标特征，计算与目标的相似度
                target_sim = None
                if self.target_features is not None:
                    target_np = self.target_features.cpu().numpy()
                    target_sim = np.dot(attacked_feat, target_np) / (
                        np.linalg.norm(attacked_feat) * np.linalg.norm(target_np) + 1e-8
                    )

                results['success'] += 1
                detail = {
                    'image': img_path.name,
                    'feature_diff': float(feat_diff),
                    'cosine_sim': float(cosine_sim)
                }
                if target_sim is not None:
                    detail['target_similarity'] = float(target_sim)
                results['details'].append(detail)

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

    优化：结合全局热度和序列末尾位置权重
    """
    data_dir = SCRIPT_DIR / 'data' / split
    sequential_path = data_dir / 'sequential_data.txt'

    if not sequential_path.exists():
        return None

    # 统计商品出现在用户序列末尾的频率（这些是被推荐/购买的商品）
    end_item_counts = Counter()
    global_item_counts = Counter()

    with open(sequential_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) > 1:
                items = parts[1:]
                for item in items:
                    global_item_counts[item] += 1
                # 序列末尾的商品更可能是被推荐的
                end_items = parts[-3:] if len(parts) > 3 else parts[1:]
                for item in end_items:
                    end_item_counts[item] += 1

    # 综合评分：末尾频率 * 2 + 全局频率
    combined_scores = {}
    for item in set(list(end_item_counts.keys()) + list(global_item_counts.keys())):
        combined_scores[item] = end_item_counts.get(item, 0) * 2 + global_item_counts.get(item, 0)

    # 获取综合得分最高的商品
    top_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:50]

    # 加载这些商品的特征
    features = []
    weights = []
    for item_id, score in top_items:
        feat_path = feature_dir / f"{item_id}.npy"
        if feat_path.exists():
            features.append(np.load(feat_path))
            weights.append(score)

    if len(features) < 5:
        return None

    # 加权平均特征 - 分数越高的商品贡献越大
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()
    features_arr = np.array(features)
    avg_feature = np.average(features_arr, axis=0, weights=weights)

    print(f"Analyzed {len(features)} high-rank items for target feature (weighted)")
    return avg_feature


def compute_per_model_targets(attacker, feature_dir: Path, split: str) -> List[np.ndarray]:
    """
    为每个CLIP模型单独计算目标特征

    这样集成攻击时每个模型都有精确的目标，而不是共享同一个目标
    """
    data_dir = SCRIPT_DIR / 'data' / split
    sequential_path = data_dir / 'sequential_data.txt'

    if not sequential_path.exists():
        return []

    # 获取热门商品图片路径
    end_item_counts = Counter()
    with open(sequential_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) > 1:
                end_items = parts[-3:] if len(parts) > 3 else parts[1:]
                for item in end_items:
                    end_item_counts[item] += 1

    top_items = [item for item, _ in end_item_counts.most_common(30)]

    # 找到这些商品的图片
    image_dir = SCRIPT_DIR / split
    per_model_features = [[] for _ in range(len(attacker.models))]

    for item_id in top_items:
        # 尝试找到对应的图片
        for ext in ['jpg', 'png']:
            img_path = image_dir / f"{item_id}.{ext}"
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert('RGB')
                    for model_idx in range(len(attacker.models)):
                        feat = attacker.extract_feature(img, model_idx=model_idx)
                        per_model_features[model_idx].append(feat)
                except Exception:
                    pass
                break

    # 计算每个模型的平均目标特征
    result = []
    for model_idx, feats in enumerate(per_model_features):
        if len(feats) > 0:
            avg = np.mean(feats, axis=0)
            result.append(avg)
            print(f"  Model {model_idx}: computed target from {len(feats)} items")
        else:
            print(f"  Model {model_idx}: no target features (will use shared)")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='VIP5 Optimized Black-box Attack',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 推荐用法 (纯目标导向 + TI-FGSM)
  python transfer_attack.py --split toys --num_images 100

  # 更大扰动 + 更多迭代
  python transfer_attack.py --split toys --num_images 100 --epsilon 0.15 --max_iter 300

  # 多方向攻击
  python transfer_attack.py --split toys --num_images 100 --attack_type multi_direction

  # 禁用集成/输入多样性 (更快但效果可能较差)
  python transfer_attack.py --split toys --num_images 100 --no_ensemble --no_diversity

参数建议:
  - epsilon: 0.10-0.15 (黑盒攻击需要更大扰动)
  - max_iter: 100-300 (更多迭代提高效果)
  - momentum: 0.9 (MI-FGSM标准值)
  - ti_kernel_size: 5-7 (TI-FGSM高斯核大小)
        """
    )

    parser.add_argument('--split', type=str, default='toys',
                        help='数据集名称 (默认: toys)')
    parser.add_argument('--num_images', type=int, default=None,
                        help='攻击图像数量 (默认: 全部)')
    parser.add_argument('--epsilon', type=float, default=0.12,
                        help='最大扰动幅度 (默认: 0.12)')
    parser.add_argument('--max_iter', type=int, default=200,
                        help='迭代次数 (默认: 200)')
    parser.add_argument('--step_size', type=float, default=None,
                        help='步长 (默认: 2.5*epsilon/max_iter)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='动量系数 (默认: 0.9)')
    parser.add_argument('--attack_type', type=str, default='toward_target',
                        choices=['toward_target', 'multi_direction'],
                        help='攻击类型 (默认: toward_target)')
    parser.add_argument('--ti_kernel_size', type=int, default=5,
                        help='TI-FGSM高斯核大小 (默认: 5)')
    parser.add_argument('--ti_sigma', type=float, default=1.0,
                        help='TI-FGSM高斯核sigma (默认: 1.0)')
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
    print("VIP5 Optimized Black-box Attack")
    print("="*60)
    print(f"Input Directory: {image_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Attack Type: {args.attack_type}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Max Iterations: {args.max_iter}")
    print(f"Momentum: {args.momentum}")
    print(f"TI-FGSM Kernel: {args.ti_kernel_size}x{args.ti_kernel_size}")
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
        attack_type=args.attack_type,
        ti_kernel_size=args.ti_kernel_size,
        ti_sigma=args.ti_sigma
    )

    # 加载目标特征
    if feature_dir.exists():
        # 分析高排名商品特征（加权版）
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

        # 尝试计算每个模型的独立目标特征
        if not args.no_ensemble:
            print("\nComputing per-model target features...")
            per_model_targets = compute_per_model_targets(attacker, feature_dir, args.split)
            if len(per_model_targets) == len(attacker.models):
                attacker.set_target_features_per_model(per_model_targets)

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

        target_sims = [d['target_similarity'] for d in results['details'] if 'target_similarity' in d]
        if target_sims:
            print(f"\nTarget Similarity Statistics:")
            print(f"  Avg Target Similarity: {np.mean(target_sims):.4f}")
            print(f"  Max Target Similarity: {np.max(target_sims):.4f}")

    # 保存结果
    result_path = output_dir / 'attack_results.json'
    with open(result_path, 'w') as f:
        json.dump({
            'method': 'optimized_transfer',
            'attack_type': args.attack_type,
            'epsilon': args.epsilon,
            'max_iter': args.max_iter,
            'momentum': args.momentum,
            'ti_kernel_size': args.ti_kernel_size,
            'ti_sigma': args.ti_sigma,
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
