#!/usr/bin/env python3
"""
VIP5 Black-box Transfer Attack v5
===================================
针对VIP5推荐系统的黑盒对抗攻击

关键改进：
1. 正确的ID映射：数字ID → ASIN → 图片文件名 → 特征文件
2. 保存224x224图片 - 避免双重缩放销毁对抗扰动
3. 逐商品最近邻目标 - 攻击向最近的热门商品靠拢
4. MI-FGSM with momentum for stable gradients

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


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


class TransferAttacker:
    """
    V5 转移攻击器

    核心设计：
    1. 只用ViT-B/32（匹配VIP5和evaluate_attack.py）
    2. 用CLIP方式处理图片做可微分攻击
    3. 保存224x224图片，避免双重缩放销毁扰动
    4. 逐商品最近邻目标
    """

    def __init__(self, device='cuda', epsilon=0.15, max_iter=200,
                 step_size=None, momentum=0.9):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.step_size = step_size if step_size else 2.0 * epsilon / max_iter
        self.momentum = momentum

        # 热门商品特征池
        self.popular_features_tensor = None

        # 加载CLIP模型
        print(f"Loading CLIP ViT-B/32 on {self.device}...")
        self.model, self.preprocess = clip.load('ViT-B/32', device=self.device)
        self.model.eval()

        # CLIP normalize参数
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)

        print("Ready.")

    def _clip_normalize(self, tensor):
        """CLIP标准化（可微分）"""
        return (tensor - self.clip_mean) / self.clip_std

    def _pil_to_tensor(self, img):
        """
        用CLIP的方式处理图片到224x224 tensor

        复刻CLIP preprocess: Resize(224, BICUBIC) + CenterCrop(224) + ToTensor
        """
        # CLIP方式: Resize最短边到224 + CenterCrop(224)
        w, h = img.size
        scale = 224.0 / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)

        # CenterCrop
        left = (new_w - 224) // 2
        top = (new_h - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))

        return transforms.ToTensor()(img).unsqueeze(0).to(self.device)

    def extract_feature(self, image):
        """用CLIP官方preprocess提取特征（和evaluate_attack.py完全一致）"""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_input)
        return features.cpu().numpy().squeeze().astype(np.float32)

    def set_popular_features(self, features):
        """设置热门商品特征池"""
        stacked = np.stack(features, axis=0)
        self.popular_features_tensor = torch.from_numpy(stacked).float().to(self.device)
        print(f"Target pool: {len(features)} popular items")

    def _find_target(self, original_feat):
        """为当前商品找最近邻热门商品作为攻击目标"""
        if self.popular_features_tensor is None:
            return None

        feat = torch.from_numpy(original_feat).float().to(self.device).unsqueeze(0)
        feat_n = F.normalize(feat, p=2, dim=1)
        pop_n = F.normalize(self.popular_features_tensor, p=2, dim=1)
        sims = torch.mm(feat_n, pop_n.t()).squeeze(0)

        # 排除自身（高相似度）
        mask = sims < 0.99
        sims_masked = sims.clone()
        sims_masked[~mask] = -1.0

        k = min(3, mask.sum().item())
        if k == 0:
            return self.popular_features_tensor[sims.argmax()]

        _, idx = sims_masked.topk(k)
        top_feats = self.popular_features_tensor[idx]
        weights = F.softmax(sims_masked[idx] * 5.0, dim=0).unsqueeze(1)
        return (top_feats * weights).sum(dim=0)

    def attack_single(self, image, target_feat):
        """MI-FGSM攻击"""
        img_tensor = self._pil_to_tensor(image)
        original_tensor = img_tensor.clone()
        perturbed = img_tensor.clone()
        momentum_grad = torch.zeros_like(perturbed)

        target = target_feat.unsqueeze(0)

        for _ in range(self.max_iter):
            perturbed.requires_grad = True

            # 提取特征
            features = self.model.encode_image(self._clip_normalize(perturbed))

            # 组合损失：余弦（方向）+ L2（幅度）
            feat_n = F.normalize(features, p=2, dim=1)
            tgt_n = F.normalize(target, p=2, dim=1)
            cos_loss = 1.0 - F.cosine_similarity(feat_n, tgt_n, dim=1).mean()
            l2_loss = F.mse_loss(features, target)
            loss = 0.7 * cos_loss + 0.3 * l2_loss / (l2_loss.detach() + 1e-8)

            loss.backward()

            with torch.no_grad():
                grad = perturbed.grad
                grad = grad / (torch.abs(grad).mean(dim=[1, 2, 3], keepdim=True) + 1e-8)
                momentum_grad = self.momentum * momentum_grad + grad
                perturbed = perturbed - self.step_size * momentum_grad.sign()
                delta = torch.clamp(perturbed - original_tensor, -self.epsilon, self.epsilon)
                perturbed = torch.clamp(original_tensor + delta, 0, 1)

            perturbed = perturbed.detach()

        # 关键：返回224x224的PIL图片，不做任何缩放
        # evaluate_attack.py的CLIP preprocess在224x224输入上是恒等变换
        # 这样对抗扰动被完美保留
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
        print(f"V5 Attack: {len(image_files)} images")
        print(f"  Model: ViT-B/32 only")
        print(f"  Epsilon: {self.epsilon}, Iter: {self.max_iter}, Step: {self.step_size:.6f}")
        print(f"  Save size: 224x224 (no double-resize)")
        print(f"  Pool: {self.popular_features_tensor.shape[0] if self.popular_features_tensor is not None else 0} items")
        print(f"{'='*60}\n")

        results = {'success': 0, 'failed': 0, 'details': []}

        for img_path in tqdm(image_files, desc="Attacking"):
            try:
                original_img = Image.open(img_path).convert('RGB')
                original_feat = self.extract_feature(original_img)

                target_feat = self._find_target(original_feat)
                if target_feat is None:
                    original_img.save(output_dir / img_path.name, quality=95)
                    continue

                # 攻击
                attacked_img = self.attack_single(original_img, target_feat)

                # 关键：保存为224x224，保留对抗扰动
                attacked_img.save(output_dir / img_path.name, quality=95)

                # 验证：用CLIP官方preprocess重新提取特征（和evaluate_attack.py一致）
                attacked_feat = self.extract_feature(attacked_img)

                feat_diff = np.linalg.norm(attacked_feat - original_feat)
                target_np = target_feat.cpu().numpy()
                sim_before = np.dot(original_feat, target_np) / (
                    np.linalg.norm(original_feat) * np.linalg.norm(target_np) + 1e-8)
                sim_after = np.dot(attacked_feat, target_np) / (
                    np.linalg.norm(attacked_feat) * np.linalg.norm(target_np) + 1e-8)

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


def load_popular_features(feature_dir: Path, split: str, top_k: int = 50) -> List[np.ndarray]:
    """
    加载热门商品特征 - 修复版

    关键：使用正确的ID映射链
    数字ID → ASIN → item2img_dict → 图片文件名 → 特征文件
    """
    data_dir = SCRIPT_DIR / 'data' / split

    # 统计商品频率
    seq_path = data_dir / 'sequential_data.txt'
    if not seq_path.exists():
        print(f"Sequential data not found: {seq_path}")
        return []

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

    # 综合得分：最后位置更重要
    scores = {}
    for item in set(list(end_counts.keys()) + list(global_counts.keys())):
        scores[item] = end_counts.get(item, 0) * 2 + global_counts.get(item, 0)

    popular_items = [item for item, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    print(f"Found {len(popular_items)} items by popularity")

    # 加载映射文件
    item2img_path = data_dir / 'item2img_dict.pkl'
    datamaps_path = data_dir / 'datamaps.json'

    if not item2img_path.exists() or not datamaps_path.exists():
        print("Mapping files not found, trying direct loading...")
        return _load_features_direct(feature_dir, top_k)

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
                feat = np.load(feat_path).astype(np.float32)
                features.append(feat)
                loaded_count += 1
            except Exception:
                pass

    print(f"Loaded {len(features)} popular item features")

    if len(features) == 0:
        print("No features loaded via mapping, trying direct loading...")
        return _load_features_direct(feature_dir, top_k)

    return features


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
    parser = argparse.ArgumentParser(description='VIP5 Black-box Attack v5')
    parser.add_argument('--split', type=str, default='toys')
    parser.add_argument('--num_images', type=int, default=None)
    parser.add_argument('--epsilon', type=float, default=0.15,
                        help='Perturbation budget (default 0.15)')
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--step_size', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
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
    print("VIP5 Black-box Attack v5")
    print("="*60)
    print(f"Input: {image_dir}")
    print(f"Output: {output_dir}")
    print(f"Features: {feature_dir}")

    attacker = TransferAttacker(
        device=args.device, epsilon=args.epsilon,
        max_iter=args.max_iter, step_size=args.step_size,
        momentum=args.momentum,
    )

    if feature_dir.exists():
        feats = load_popular_features(feature_dir, args.split, args.top_k_popular)
        if feats:
            attacker.set_popular_features(feats)

    results = attacker.attack_batch(image_dir, output_dir, args.num_images)

    print(f"\n{'='*60}")
    print(f"Done! Success: {results['success']}, Failed: {results['failed']}")

    valid = [d for d in results['details'] if 'feat_diff' in d]
    if valid:
        gains = [d['sim_gain'] for d in valid]
        pos = [g for g in gains if g > 0]
        print(f"  Avg feature diff: {np.mean([d['feat_diff'] for d in valid]):.4f}")
        print(f"  Avg target sim gain: {np.mean(gains):+.4f}")
        print(f"  Moved closer to target: {len(pos)}/{len(gains)} ({100*len(pos)/len(gains):.1f}%)")

    with open(output_dir / 'attack_results.json', 'w') as f:
        json.dump({'method': 'v5', 'epsilon': args.epsilon, 'results': results}, f, indent=2)

    print(f"\nNext steps:")
    print(f"  python evaluate_attack.py --mode extract --split {args.split}")
    print(f"  python evaluate_attack_vip5.py --split {args.split} --num_samples 500")


if __name__ == '__main__':
    main()
