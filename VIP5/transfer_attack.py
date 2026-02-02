#!/usr/bin/env python3
"""
VIP5 Black-box Transfer Attack v7
===================================
针对VIP5推荐系统的黑盒迁移对抗攻击

核心改进（相比v6）：
1. TI-FGSM: 平移不变性 - 梯度高斯核平滑，大幅提高迁移性
2. SI-FGSM: 尺度不变性 - 多尺度梯度计算
3. VMI-FGSM: 方差调节 - 在邻域采样减少梯度方差
4. Admix: 混合其他图片增强多样性
5. 更强的Input Diversity范围
6. 特征空间正则化 - 同时最大化远离原始特征

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


def get_gaussian_kernel(kernel_size=15, sigma=3.0):
    """生成用于TI-FGSM的高斯核"""
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d / kernel_2d.sum()
    # shape: [1, 1, kernel_size, kernel_size], repeat for 3 channels
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
    return kernel


class EnsembleTransferAttacker:
    """
    集成迁移攻击器 v7 (Ensemble Transfer Attacker)

    技术栈：
    1. Ensemble Gradient: 多模型梯度平均
    2. MI-FGSM: 动量迭代FGSM
    3. TI-FGSM: 平移不变性 (Translation-Invariant)
    4. SI-FGSM: 尺度不变性 (Scale-Invariant)
    5. VMI-FGSM: 方差调节动量 (Variance Tuning)
    6. DI-FGSM: 输入多样性 (Input Diversity)
    7. Admix: 混合输入增强
    """

    def __init__(self, device='cuda', epsilon=0.15, max_iter=300,
                 step_size=None, momentum=1.0, diversity_prob=0.7,
                 ti_kernel_size=15, ti_sigma=3.0,
                 si_num_scales=5, vmi_beta=1.5, vmi_num_samples=10,
                 admix_ratio=0.2, admix_num=3):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.step_size = step_size if step_size else epsilon / max_iter * 2.5
        self.momentum = momentum
        self.diversity_prob = diversity_prob

        # TI-FGSM参数
        self.ti_kernel_size = ti_kernel_size
        self.gaussian_kernel = get_gaussian_kernel(ti_kernel_size, ti_sigma).to(self.device)

        # SI-FGSM参数
        self.si_num_scales = si_num_scales

        # VMI-FGSM参数
        self.vmi_beta = vmi_beta
        self.vmi_num_samples = vmi_num_samples

        # Admix参数
        self.admix_ratio = admix_ratio
        self.admix_num = admix_num
        self.admix_images = []  # 用于混合的图片

        # 热门商品特征池
        self.popular_features_tensor = None
        self.popular_image_paths = []
        self.surrogate_popular_features = []

        print(f"\n{'='*60}")
        print("Loading Surrogate Model Ensemble for Transfer Attack v7")
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
                model = model.float()
                self.surrogate_models.append(model)
                self.surrogate_preprocesses.append(preprocess)
                print(f"    ✓ {model_name} loaded")
            except Exception as e:
                print(f"    ✗ Failed to load {model_name}: {e}")

        if len(self.surrogate_models) == 0:
            raise RuntimeError("No surrogate models loaded!")

        print(f"\nLoaded {len(self.surrogate_models)} surrogate models")

        # 加载目标模型（仅用于特征验证）
        print(f"\nLoading target model {TARGET_MODEL} (for validation only)...")
        self.target_model, self.target_preprocess = clip.load(TARGET_MODEL, device=self.device)
        self.target_model.eval()
        self.target_model = self.target_model.float()

        # CLIP normalize参数
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)

        print("Ready.\n")

    def _clip_normalize(self, tensor):
        """CLIP标准化（可微分）"""
        return (tensor - self.clip_mean) / self.clip_std

    def _input_diversity(self, tensor):
        """
        Enhanced Input Diversity Transform (DI-FGSM)
        更大的resize范围 + 随机插值方法
        """
        if random.random() > self.diversity_prob:
            return tensor

        # 更大的缩放范围（160-224）提高多样性
        rnd = random.randint(160, 224)
        mode = random.choice(['bilinear', 'bicubic', 'nearest'])
        align = False if mode == 'nearest' else True
        rescaled = F.interpolate(tensor, size=(rnd, rnd), mode=mode,
                                 align_corners=align if mode != 'nearest' else None)

        h_rem = 224 - rnd
        w_rem = 224 - rnd
        pad_top = random.randint(0, h_rem)
        pad_bottom = h_rem - pad_top
        pad_left = random.randint(0, w_rem)
        pad_right = w_rem - pad_left
        padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        return padded

    def _ti_smooth_gradient(self, grad):
        """TI-FGSM: 用高斯核平滑梯度，增强平移不变性"""
        padding = self.ti_kernel_size // 2
        # depthwise convolution
        smoothed = F.conv2d(grad, self.gaussian_kernel, padding=padding, groups=3)
        return smoothed

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
        """设置热门商品特征池"""
        stacked = np.stack(features, axis=0)
        self.popular_features_tensor = torch.from_numpy(stacked).float().to(self.device)
        print(f"Target pool: {len(features)} popular items")

    def set_popular_images(self, image_paths: List[Path]):
        """设置热门商品图片路径并提取代理模型特征"""
        self.popular_image_paths = image_paths
        print(f"Popular images for surrogates: {len(image_paths)} images")

        # 预加载admix用的图片tensor
        self.admix_images = []
        for img_path in image_paths[:20]:
            try:
                img = Image.open(img_path).convert('RGB')
                tensor = self._pil_to_tensor(img)
                self.admix_images.append(tensor)
            except:
                pass
        print(f"  Loaded {len(self.admix_images)} images for Admix")

        # 预提取每个代理模型的热门商品特征
        self.surrogate_popular_features = []
        for idx, model in enumerate(self.surrogate_models):
            print(f"  Extracting features for surrogate {idx+1}/{len(self.surrogate_models)}...")
            features = []
            for img_path in image_paths[:50]:
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
        """为当前商品找最近邻热门商品的索引"""
        if self.popular_features_tensor is None:
            return None, None

        feat = torch.from_numpy(original_feat).float().to(self.device).unsqueeze(0)
        feat_n = F.normalize(feat, p=2, dim=1)
        pop_n = F.normalize(self.popular_features_tensor, p=2, dim=1)
        sims = torch.mm(feat_n, pop_n.t()).squeeze(0)

        mask = sims < 0.99
        sims_masked = sims.clone()
        sims_masked[~mask] = -1.0

        # 使用更多的目标（5个而不是3个），增加目标多样性
        k = min(5, mask.sum().item())
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

    def _compute_ensemble_loss(self, perturbed, target_indices, weights, original_tensor):
        """
        计算集成损失 v7：
        1. 多模型梯度聚合
        2. SI-FGSM: 多尺度计算
        3. Admix: 混合输入
        4. 双目标: 接近热门 + 远离原始
        """
        total_loss = 0.0
        valid_models = 0

        for idx, model in enumerate(self.surrogate_models):
            target_feat = self._get_surrogate_target(idx, target_indices, weights)
            if target_feat is None:
                continue

            model_loss = 0.0

            # SI-FGSM: 多尺度
            for si in range(self.si_num_scales):
                scale = 1.0 / (2 ** si)
                if scale < 1.0:
                    scaled = F.interpolate(perturbed, scale_factor=scale, mode='bilinear', align_corners=False)
                    scaled = F.interpolate(scaled, size=(224, 224), mode='bilinear', align_corners=False)
                else:
                    scaled = perturbed

                # Input Diversity
                diverse_input = self._input_diversity(scaled)
                normalized = self._clip_normalize(diverse_input)

                # 提取特征
                features = model.encode_image(normalized)
                feat_n = F.normalize(features, p=2, dim=1)
                tgt_n = F.normalize(target_feat.unsqueeze(0), p=2, dim=1)

                # 目标损失：最大化与热门商品的相似度
                cos_loss = 1.0 - F.cosine_similarity(feat_n, tgt_n, dim=1).mean()
                model_loss += cos_loss

                # Admix: 混合其他图片增强多样性
                if self.admix_images and si == 0 and random.random() < 0.5:
                    for _ in range(self.admix_num):
                        mix_img = random.choice(self.admix_images)
                        mixed = perturbed * (1 - self.admix_ratio) + mix_img * self.admix_ratio
                        mixed_div = self._input_diversity(mixed)
                        mixed_norm = self._clip_normalize(mixed_div)
                        mixed_feat = model.encode_image(mixed_norm)
                        mixed_feat_n = F.normalize(mixed_feat, p=2, dim=1)
                        mix_loss = 1.0 - F.cosine_similarity(mixed_feat_n, tgt_n, dim=1).mean()
                        model_loss += mix_loss * 0.2

            # 远离原始特征的正则化损失
            orig_norm = self._clip_normalize(original_tensor)
            with torch.no_grad():
                orig_feat = model.encode_image(orig_norm)
            pert_norm = self._clip_normalize(perturbed)
            pert_feat = model.encode_image(pert_norm)
            orig_feat_n = F.normalize(orig_feat, p=2, dim=1)
            pert_feat_n = F.normalize(pert_feat, p=2, dim=1)
            repel_loss = F.cosine_similarity(pert_feat_n, orig_feat_n, dim=1).mean()
            model_loss += repel_loss * 0.3

            total_loss += model_loss / self.si_num_scales
            valid_models += 1

        if valid_models == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return total_loss / valid_models

    def attack_single(self, image, target_indices, weights):
        """
        集成迁移攻击 v7: MI-TI-SI-VMI-FGSM + Admix
        """
        img_tensor = self._pil_to_tensor(image)
        original_tensor = img_tensor.clone()
        perturbed = img_tensor.clone()
        momentum_grad = torch.zeros_like(perturbed)
        variance = torch.zeros_like(perturbed)

        for iteration in range(self.max_iter):
            perturbed.requires_grad = True

            loss = self._compute_ensemble_loss(perturbed, target_indices, weights, original_tensor)
            loss.backward()

            with torch.no_grad():
                grad = perturbed.grad.clone()

                # TI-FGSM: 高斯核平滑梯度
                grad = self._ti_smooth_gradient(grad)

                # VMI-FGSM: 方差调节
                if iteration > 0:
                    # 在邻域采样计算梯度方差
                    sample_grads = []
                    for _ in range(self.vmi_num_samples):
                        neighbor = perturbed.detach() + torch.randn_like(perturbed) * (self.vmi_beta * self.epsilon)
                        neighbor = torch.clamp(neighbor, 0, 1)
                        neighbor.requires_grad = True
                        n_loss = self._compute_ensemble_loss(neighbor, target_indices, weights, original_tensor)
                        n_loss.backward()
                        sample_grads.append(neighbor.grad.clone())
                        neighbor.requires_grad = False

                    # 计算邻域梯度均值
                    avg_neighbor_grad = torch.stack(sample_grads).mean(dim=0)
                    avg_neighbor_grad = self._ti_smooth_gradient(avg_neighbor_grad)
                    # 结合当前梯度和邻域梯度
                    grad = grad + variance
                    variance = avg_neighbor_grad - grad

                # MI-FGSM: 归一化 + 动量
                grad = grad / (torch.abs(grad).mean(dim=[1, 2, 3], keepdim=True) + 1e-8)
                momentum_grad = self.momentum * momentum_grad + grad

                # 更新
                perturbed = perturbed - self.step_size * momentum_grad.sign()
                delta = torch.clamp(perturbed - original_tensor, -self.epsilon, self.epsilon)
                perturbed = torch.clamp(original_tensor + delta, 0, 1)

            perturbed = perturbed.detach()

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
        print(f"Ensemble Transfer Attack v7")
        print(f"{'='*60}")
        print(f"  Target (VIP5): {TARGET_MODEL} - NOT used in optimization")
        print(f"  Surrogates: {', '.join(SURROGATE_MODELS)}")
        print(f"  Images: {len(image_files)}")
        print(f"  Epsilon: {self.epsilon}, Iter: {self.max_iter}")
        print(f"  Techniques: MI + TI + SI + VMI + DI + Admix")
        print(f"  Input Diversity: {self.diversity_prob*100:.0f}%")
        print(f"  SI scales: {self.si_num_scales}")
        print(f"  TI kernel: {self.ti_kernel_size}")
        print(f"  VMI samples: {self.vmi_num_samples}, beta: {self.vmi_beta}")
        print(f"  Pool: {self.popular_features_tensor.shape[0] if self.popular_features_tensor is not None else 0} items")
        print(f"{'='*60}\n")

        results = {'success': 0, 'failed': 0, 'details': []}

        for img_path in tqdm(image_files, desc="Transfer Attacking"):
            try:
                original_img = Image.open(img_path).convert('RGB')
                original_feat = self.extract_feature_target(original_img)

                target_indices, weights = self._find_target_indices(original_feat)
                if target_indices is None:
                    original_img.save(output_dir / img_path.name, quality=95)
                    continue

                attacked_img = self.attack_single(original_img, target_indices, weights)
                attacked_img.save(output_dir / img_path.name, quality=95)

                # 用目标模型验证
                attacked_feat = self.extract_feature_target(attacked_img)

                feat_diff = np.linalg.norm(attacked_feat - original_feat)
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
                import traceback
                traceback.print_exc()
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
    parser = argparse.ArgumentParser(description='VIP5 Ensemble Transfer Attack v7')
    parser.add_argument('--split', type=str, default='toys')
    parser.add_argument('--num_images', type=int, default=None)
    parser.add_argument('--epsilon', type=float, default=0.15,
                        help='Perturbation budget (default 0.15)')
    parser.add_argument('--max_iter', type=int, default=300)
    parser.add_argument('--step_size', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=1.0)
    parser.add_argument('--diversity_prob', type=float, default=0.7,
                        help='Input diversity probability (default 0.7)')
    parser.add_argument('--top_k_popular', type=int, default=50)
    parser.add_argument('--ti_kernel', type=int, default=15,
                        help='TI-FGSM Gaussian kernel size')
    parser.add_argument('--si_scales', type=int, default=5,
                        help='SI-FGSM number of scales')
    parser.add_argument('--vmi_samples', type=int, default=10,
                        help='VMI-FGSM neighborhood samples')
    parser.add_argument('--vmi_beta', type=float, default=1.5,
                        help='VMI-FGSM neighborhood radius factor')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    image_dir = SCRIPT_DIR / args.split
    output_dir = SCRIPT_DIR / f'{args.split}2'
    feature_dir = SCRIPT_DIR / 'features' / 'vitb32_features' / f'{args.split}_original'

    if not image_dir.exists():
        print(f"Error: {image_dir} not found")
        sys.exit(1)

    print("\n" + "="*60)
    print("VIP5 Ensemble Transfer Attack v7")
    print("="*60)
    print(f"Input: {image_dir}")
    print(f"Output: {output_dir}")
    print(f"Features: {feature_dir}")
    print()
    print("Transfer Attack Setup:")
    print(f"  Target Model (VIP5): {TARGET_MODEL}")
    print(f"  Surrogate Models: {SURROGATE_MODELS}")
    print(f"  Techniques: MI + TI + SI + VMI + DI + Admix")
    print(f"  (Surrogates ≠ Target → True Transfer Attack)")
    print("="*60)

    attacker = EnsembleTransferAttacker(
        device=args.device, epsilon=args.epsilon,
        max_iter=args.max_iter, step_size=args.step_size,
        momentum=args.momentum, diversity_prob=args.diversity_prob,
        ti_kernel_size=args.ti_kernel, si_num_scales=args.si_scales,
        vmi_num_samples=args.vmi_samples, vmi_beta=args.vmi_beta,
    )

    original_image_dir = SCRIPT_DIR / f'{args.split}_original'
    if not original_image_dir.exists():
        original_image_dir = image_dir

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
            'method': 'ensemble_transfer_v7',
            'target_model': TARGET_MODEL,
            'surrogate_models': SURROGATE_MODELS,
            'epsilon': args.epsilon,
            'diversity_prob': args.diversity_prob,
            'techniques': ['MI-FGSM', 'TI-FGSM', 'SI-FGSM', 'VMI-FGSM', 'DI-FGSM', 'Admix'],
            'results': results
        }, f, indent=2)

    print(f"\nNext steps:")
    print(f"  python evaluate_attack.py --mode extract --split {args.split}")
    print(f"  python evaluate_attack_vip5.py --split {args.split} --num_samples 500")


if __name__ == '__main__':
    main()
