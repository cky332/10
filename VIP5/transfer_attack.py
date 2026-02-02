#!/usr/bin/env python3
"""
VIP5 Black-box Transfer Attack v6
===================================
针对VIP5推荐系统的黑盒迁移对抗攻击

核心设计 - 真正的迁移攻击：
1. 使用不同于目标模型的代理模型集成（Surrogate Ensemble）
   - VIP5目标模型: CLIP ViT-B/32
   - 代理模型: CLIP RN50, RN101, ViT-L/14 (故意不包含ViT-B/32)
2. 集成梯度：聚合多个代理模型的梯度，提高迁移性
3. MI-FGSM + Input Diversity：增强对抗样本的迁移能力

迁移攻击原理：
  在代理模型上生成对抗样本 → 希望扰动能迁移到目标模型
  这体现了黑盒攻击的本质：攻击者无法访问目标模型

使用方法：
    python transfer_attack.py --split toys
    python transfer_attack.py --split toys --num_images 100 --epsilon 0.15
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
from typing import List, Dict, Optional, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import clip

SCRIPT_DIR = Path(__file__).resolve().parent

# 代理模型配置（故意不包含VIP5使用的ViT-B/32，体现迁移攻击）
SURROGATE_MODELS = ['RN50', 'RN101', 'ViT-L/14']
# 目标模型（VIP5使用的模型，仅用于特征提取验证，不参与攻击）
TARGET_MODEL = 'ViT-B/32'


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


class EnsembleTransferAttacker:
    """
    集成迁移攻击器 (Ensemble Transfer Attacker)

    核心思想：
    - 使用多个不同架构的代理模型（RN50, RN101, ViT-L/14）
    - 故意排除目标模型（ViT-B/32）以体现真正的迁移攻击
    - 聚合多个模型的梯度来生成更具迁移性的对抗样本

    技术：
    1. Ensemble Gradient: 多模型梯度平均
    2. MI-FGSM: 动量迭代FGSM，稳定梯度方向
    3. Input Diversity: 输入多样性变换，增强迁移性
    """

    def __init__(self, device='cuda', epsilon=0.15, max_iter=200,
                 step_size=None, momentum=0.9, diversity_prob=0.5):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.step_size = step_size if step_size else 2.0 * epsilon / max_iter
        self.momentum = momentum
        self.diversity_prob = diversity_prob  # Input Diversity概率

        # 热门商品特征池（用目标模型特征）
        self.popular_features_tensor = None
        self.popular_image_paths = []
        self.surrogate_popular_features = []

        # 加载代理模型集成（用于生成对抗样本）
        print(f"\n{'='*60}")
        print("Loading Surrogate Model Ensemble for Transfer Attack")
        print(f"{'='*60}")
        print(f"Target Model (VIP5): {TARGET_MODEL} - NOT used for attack")
        print(f"Surrogate Models: {SURROGATE_MODELS}")
        print(f"{'='*60}\n")

        self.surrogate_models = []
        self.surrogate_preprocesses = []

        for model_name in SURROGATE_MODELS:
            print(f"  Loading surrogate: {model_name}...")
            try:
                model, preprocess = clip.load(model_name, device=self.device)
                model.eval()
                model = model.float()  # 转换为float32
                self.surrogate_models.append(model)
                self.surrogate_preprocesses.append(preprocess)
                print(f"    ✓ {model_name} loaded")
            except Exception as e:
                print(f"    ✗ Failed to load {model_name}: {e}")

        if len(self.surrogate_models) == 0:
            raise RuntimeError("No surrogate models loaded!")

        print(f"\nLoaded {len(self.surrogate_models)} surrogate models")

        # 加载目标模型（仅用于特征验证，不参与攻击优化）
        print(f"\nLoading target model {TARGET_MODEL} (for validation only)...")
        self.target_model, self.target_preprocess = clip.load(TARGET_MODEL, device=self.device)
        self.target_model.eval()
        self.target_model = self.target_model.float()

        # CLIP normalize参数（所有CLIP模型共享）
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)

        print("Ready.\n")

    def _clip_normalize(self, tensor):
        """CLIP标准化（可微分）"""
        return (tensor - self.clip_mean) / self.clip_std

    def _input_diversity(self, tensor):
        """
        Input Diversity Transform (DI-FGSM)
        随机resize和padding，增强对抗样本的迁移性
        """
        if random.random() > self.diversity_prob:
            return tensor

        # 随机缩放
        rnd = random.randint(200, 224)
        rescaled = F.interpolate(tensor, size=(rnd, rnd), mode='bilinear', align_corners=False)

        # 随机padding回224
        h_rem = 224 - rnd
        w_rem = 224 - rnd
        pad_top = random.randint(0, h_rem)
        pad_bottom = h_rem - pad_top
        pad_left = random.randint(0, w_rem)
        pad_right = w_rem - pad_left
        padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        return padded

    def _pil_to_tensor(self, img, size=224):
        """用CLIP的方式处理图片到指定尺寸tensor"""
        w, h = img.size
        scale = float(size) / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)

        left = (new_w - size) // 2
        top = (new_h - size) // 2
        img = img.crop((left, top, left + size, top + size))

        return transforms.ToTensor()(img).unsqueeze(0).to(self.device)

    def extract_feature_target(self, image):
        """用目标模型提取特征（和evaluate_attack.py一致）"""
        image_input = self.target_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.target_model.encode_image(image_input)
        return features.cpu().numpy().squeeze().astype(np.float32)

    def set_popular_features(self, features):
        """设置热门商品特征池（用于目标模型验证）"""
        stacked = np.stack(features, axis=0)
        self.popular_features_tensor = torch.from_numpy(stacked).float().to(self.device)
        print(f"Target pool: {len(features)} popular items")

    def set_popular_images(self, image_paths: List[Path]):
        """
        设置热门商品图片路径
        每个代理模型会提取自己的特征表示
        """
        self.popular_image_paths = image_paths
        print(f"Popular images for surrogates: {len(image_paths)} images")

        # 预提取每个代理模型的热门商品特征
        self.surrogate_popular_features = []
        for idx, model in enumerate(self.surrogate_models):
            print(f"  Extracting features for surrogate {idx+1}/{len(self.surrogate_models)}...")
            features = []
            for img_path in image_paths[:50]:  # 限制数量
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_input = self.surrogate_preprocesses[idx](img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        feat = model.encode_image(img_input)
                    features.append(feat.squeeze(0))
                except:
                    pass
            if features:
                self.surrogate_popular_features.append(torch.stack(features))
            else:
                self.surrogate_popular_features.append(None)
        print(f"  Done extracting surrogate features")

    def _find_target_indices(self, original_feat):
        """为当前商品找最近邻热门商品的索引（用于所有模型）"""
        if self.popular_features_tensor is None:
            return None, None

        feat = torch.from_numpy(original_feat).float().to(self.device).unsqueeze(0)
        feat_n = F.normalize(feat, p=2, dim=1)
        pop_n = F.normalize(self.popular_features_tensor, p=2, dim=1)
        sims = torch.mm(feat_n, pop_n.t()).squeeze(0)

        mask = sims < 0.99
        sims_masked = sims.clone()
        sims_masked[~mask] = -1.0

        k = min(3, mask.sum().item())
        if k == 0:
            idx = sims.argmax().unsqueeze(0)
            weights = torch.ones(1, device=self.device)
        else:
            _, idx = sims_masked.topk(k)
            weights = F.softmax(sims_masked[idx] * 5.0, dim=0)

        return idx, weights

    def _get_surrogate_target(self, surrogate_idx, target_indices, weights):
        """获取特定代理模型的目标特征"""
        if self.surrogate_popular_features[surrogate_idx] is None:
            return None
        pop_feats = self.surrogate_popular_features[surrogate_idx]
        target_feats = pop_feats[target_indices]
        return (target_feats * weights.unsqueeze(1)).sum(dim=0)

    def _compute_ensemble_loss(self, perturbed, target_indices, weights):
        """
        计算集成损失：聚合所有代理模型的损失

        这是迁移攻击的核心：在多个不同的代理模型上优化
        每个代理模型使用自己的特征空间计算目标
        """
        total_loss = 0.0
        valid_models = 0

        for idx, model in enumerate(self.surrogate_models):
            # 获取该代理模型的目标特征
            target_feat = self._get_surrogate_target(idx, target_indices, weights)
            if target_feat is None:
                continue

            # 应用Input Diversity
            diverse_input = self._input_diversity(perturbed)
            normalized = self._clip_normalize(diverse_input)

            # 提取代理模型特征
            features = model.encode_image(normalized)

            # 计算损失（在该模型自己的特征空间内）
            feat_n = F.normalize(features, p=2, dim=1)
            tgt_n = F.normalize(target_feat.unsqueeze(0), p=2, dim=1)

            cos_loss = 1.0 - F.cosine_similarity(feat_n, tgt_n, dim=1).mean()
            total_loss += cos_loss
            valid_models += 1

        if valid_models == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # 平均损失
        return total_loss / valid_models

    def attack_single(self, image, target_indices, weights):
        """
        集成迁移攻击 (Ensemble Transfer Attack with MI-FGSM)

        关键：使用代理模型集成生成对抗样本，不使用目标模型
        每个代理模型在自己的特征空间内优化
        """
        img_tensor = self._pil_to_tensor(image)
        original_tensor = img_tensor.clone()
        perturbed = img_tensor.clone()
        momentum_grad = torch.zeros_like(perturbed)

        for _ in range(self.max_iter):
            perturbed.requires_grad = True

            # 计算集成损失（在所有代理模型上，使用各自的目标特征）
            loss = self._compute_ensemble_loss(perturbed, target_indices, weights)

            loss.backward()

            with torch.no_grad():
                grad = perturbed.grad

                # MI-FGSM: 归一化梯度 + 动量
                grad = grad / (torch.abs(grad).mean(dim=[1, 2, 3], keepdim=True) + 1e-8)
                momentum_grad = self.momentum * momentum_grad + grad

                # 更新扰动
                perturbed = perturbed - self.step_size * momentum_grad.sign()
                delta = torch.clamp(perturbed - original_tensor, -self.epsilon, self.epsilon)
                perturbed = torch.clamp(original_tensor + delta, 0, 1)

            perturbed = perturbed.detach()

        # 返回224x224的PIL图片
        result = perturbed.squeeze(0).cpu()
        result = torch.clamp(result, 0, 1)
        return transforms.ToPILImage()(result)

    def attack_batch(self, image_dir, output_dir, num_images=None):
        """批量攻击"""
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        if num_images and num_images < len(image_files):
            random.seed(42)
            image_files = random.sample(image_files, num_images)

        print(f"\n{'='*60}")
        print(f"Ensemble Transfer Attack v6")
        print(f"{'='*60}")
        print(f"  Target (VIP5): {TARGET_MODEL} - NOT used in optimization")
        print(f"  Surrogates: {', '.join(SURROGATE_MODELS)}")
        print(f"  Images: {len(image_files)}")
        print(f"  Epsilon: {self.epsilon}, Iter: {self.max_iter}")
        print(f"  Input Diversity: {self.diversity_prob*100:.0f}%")
        print(f"  Pool: {self.popular_features_tensor.shape[0] if self.popular_features_tensor is not None else 0} items")
        print(f"{'='*60}\n")

        results = {'success': 0, 'failed': 0, 'details': []}

        for img_path in tqdm(image_files, desc="Transfer Attacking"):
            try:
                original_img = Image.open(img_path).convert('RGB')
                # 用目标模型提取原始特征（用于找最近邻和验证）
                original_feat = self.extract_feature_target(original_img)

                # 找目标索引（在目标模型特征空间中）
                target_indices, weights = self._find_target_indices(original_feat)
                if target_indices is None:
                    original_img.save(output_dir / img_path.name, quality=95)
                    continue

                # 在代理模型上执行攻击（每个模型用自己的特征空间）
                attacked_img = self.attack_single(original_img, target_indices, weights)

                # 保存224x224
                attacked_img.save(output_dir / img_path.name, quality=95)

                # 用目标模型验证攻击效果
                attacked_feat = self.extract_feature_target(attacked_img)

                feat_diff = np.linalg.norm(attacked_feat - original_feat)
                # 用目标模型的热门特征计算相似度变化
                target_feat_np = self.popular_features_tensor[target_indices].cpu().numpy()
                target_feat_weighted = (target_feat_np * weights.cpu().numpy()[:, np.newaxis]).sum(axis=0)
                sim_before = np.dot(original_feat, target_feat_weighted) / (
                    np.linalg.norm(original_feat) * np.linalg.norm(target_feat_weighted) + 1e-8)
                sim_after = np.dot(attacked_feat, target_feat_weighted) / (
                    np.linalg.norm(attacked_feat) * np.linalg.norm(target_feat_weighted) + 1e-8)

                results['success'] += 1
                results['details'].append({
                    'image': img_path.name,
                    'feat_diff': float(feat_diff),
                    'sim_before': float(sim_before),
                    'sim_after': float(sim_after),
                    'sim_gain': float(sim_after - sim_before),
                })

            except Exception as e:
                print(f"\nError: {img_path.name}: {e}")
                results['failed'] += 1

        return results


def load_popular_features_and_images(feature_dir: Path, image_dir: Path, split: str, top_k: int = 50) -> Tuple[List[np.ndarray], List[Path]]:
    """加载热门商品特征和图片路径"""
    data_dir = SCRIPT_DIR / 'data' / split

    seq_path = data_dir / 'sequential_data.txt'
    if not seq_path.exists():
        print(f"Sequential data not found: {seq_path}")
        return [], []

    end_counts = Counter()
    global_counts = Counter()
    with open(seq_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) > 1:
                for item in parts[1:]:
                    global_counts[item] += 1
                for item in (parts[-3:] if len(parts) > 3 else parts[1:]):
                    end_counts[item] += 1

    scores = {}
    for item in set(list(end_counts.keys()) + list(global_counts.keys())):
        scores[item] = end_counts.get(item, 0) * 2 + global_counts.get(item, 0)

    popular_items = [item for item, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    print(f"Found {len(popular_items)} items by popularity")

    item2img_path = data_dir / 'item2img_dict.pkl'
    datamaps_path = data_dir / 'datamaps.json'

    if not item2img_path.exists() or not datamaps_path.exists():
        print("Mapping files not found, trying direct loading...")
        return _load_features_direct(feature_dir, top_k), []

    item2img_dict = load_pickle(str(item2img_path))
    datamaps = load_json(str(datamaps_path))
    id2item = datamaps.get('id2item', {})

    id_to_imgname = {}
    for asin_key, img_info in item2img_dict.items():
        if isinstance(img_info, str):
            img_filename = img_info
        elif isinstance(img_info, list) and len(img_info) > 0:
            img_filename = img_info[0]
        else:
            continue

        if '/' in img_filename:
            img_filename = img_filename.split('/')[-1]
        if '.' in img_filename:
            img_filename = img_filename.rsplit('.', 1)[0]

        for num_id, asin in id2item.items():
            if asin == asin_key:
                id_to_imgname[num_id] = img_filename
                break

    print(f"Built ID to image name mapping: {len(id_to_imgname)} entries")

    features = []
    image_paths = []
    loaded_count = 0

    for item_id in popular_items:
        if loaded_count >= top_k:
            break

        img_name = id_to_imgname.get(str(item_id))
        if img_name is None:
            continue

        feat_path = feature_dir / f"{img_name}.npy"
        # 查找对应的图片文件
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = image_dir / f"{img_name}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if feat_path.exists() and img_path is not None:
            try:
                feat = np.load(feat_path).astype(np.float32)
                features.append(feat)
                image_paths.append(img_path)
                loaded_count += 1
            except Exception:
                pass

    print(f"Loaded {len(features)} popular item features and images")

    if len(features) == 0:
        print("No features loaded via mapping, trying direct loading...")
        return _load_features_direct(feature_dir, top_k), []

    return features, image_paths


def _load_features_direct(feature_dir: Path, top_k: int = 50) -> List[np.ndarray]:
    """直接从特征目录加载特征（备用方案）"""
    if not feature_dir.exists():
        return []

    feature_files = list(feature_dir.glob('*.npy'))
    print(f"Direct loading from {len(feature_files)} feature files")

    if len(feature_files) == 0:
        return []

    random.seed(42)
    if len(feature_files) > top_k:
        selected_files = random.sample(feature_files, top_k)
    else:
        selected_files = feature_files

    features = []
    for f in selected_files:
        try:
            feat = np.load(f).astype(np.float32)
            features.append(feat)
        except Exception:
            pass

    print(f"Loaded {len(features)} features directly")
    return features


def main():
    parser = argparse.ArgumentParser(description='VIP5 Ensemble Transfer Attack v6')
    parser.add_argument('--split', type=str, default='toys')
    parser.add_argument('--num_images', type=int, default=None)
    parser.add_argument('--epsilon', type=float, default=0.15,
                        help='Perturbation budget (default 0.15)')
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--step_size', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--diversity_prob', type=float, default=0.5,
                        help='Input diversity probability (default 0.5)')
    parser.add_argument('--top_k_popular', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    image_dir = SCRIPT_DIR / args.split
    output_dir = SCRIPT_DIR / f'{args.split}2'
    feature_dir = SCRIPT_DIR / 'features' / 'vitb32_features' / f'{args.split}_original'

    if not image_dir.exists():
        print(f"Error: {image_dir} not found")
        sys.exit(1)

    print("\n" + "="*60)
    print("VIP5 Ensemble Transfer Attack v6")
    print("="*60)
    print(f"Input: {image_dir}")
    print(f"Output: {output_dir}")
    print(f"Features: {feature_dir}")
    print()
    print("Transfer Attack Setup:")
    print(f"  Target Model (VIP5): {TARGET_MODEL}")
    print(f"  Surrogate Models: {SURROGATE_MODELS}")
    print(f"  (Surrogates ≠ Target → True Transfer Attack)")
    print("="*60)

    attacker = EnsembleTransferAttacker(
        device=args.device, epsilon=args.epsilon,
        max_iter=args.max_iter, step_size=args.step_size,
        momentum=args.momentum, diversity_prob=args.diversity_prob,
    )

    # 加载原始图片目录（用于提取代理模型的热门商品特征）
    original_image_dir = SCRIPT_DIR / f'{args.split}_original'
    if not original_image_dir.exists():
        original_image_dir = image_dir  # 如果没有原始目录，用当前目录

    if feature_dir.exists():
        feats, img_paths = load_popular_features_and_images(
            feature_dir, original_image_dir, args.split, args.top_k_popular
        )
        if feats:
            attacker.set_popular_features(feats)
        if img_paths:
            attacker.set_popular_images(img_paths)

    results = attacker.attack_batch(image_dir, output_dir, args.num_images)

    print(f"\n{'='*60}")
    print(f"Transfer Attack Complete!")
    print(f"{'='*60}")
    print(f"Success: {results['success']}, Failed: {results['failed']}")

    valid = [d for d in results['details'] if 'feat_diff' in d]
    if valid:
        gains = [d['sim_gain'] for d in valid]
        pos = [g for g in gains if g > 0]
        print(f"\nTransfer Attack Results (on target model {TARGET_MODEL}):")
        print(f"  Avg feature diff: {np.mean([d['feat_diff'] for d in valid]):.4f}")
        print(f"  Avg target sim gain: {np.mean(gains):+.4f}")
        print(f"  Successful transfers: {len(pos)}/{len(gains)} ({100*len(pos)/len(gains):.1f}%)")

    with open(output_dir / 'attack_results.json', 'w') as f:
        json.dump({
            'method': 'ensemble_transfer_v6',
            'target_model': TARGET_MODEL,
            'surrogate_models': SURROGATE_MODELS,
            'epsilon': args.epsilon,
            'diversity_prob': args.diversity_prob,
            'results': results
        }, f, indent=2)

    print(f"\nNext steps:")
    print(f"  python evaluate_attack.py --mode extract --split {args.split}")
    print(f"  python evaluate_attack_vip5.py --split {args.split} --num_samples 500")


if __name__ == '__main__':
    main()
