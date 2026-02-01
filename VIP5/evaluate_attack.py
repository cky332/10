#!/usr/bin/env python3
"""
VIP5 Typography Attack Evaluation
==================================
对比原始图片和攻击后图片对推荐结果的影响

Usage:
    python evaluate_attack.py --split toys
    python evaluate_attack.py --mode compare --split toys --top_k 10
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
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / 'src'))

import torch
import clip


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class CLIPFeatureExtractor:
    """CLIP特征提取器"""
    
    def __init__(self, model_name: str = 'ViT-B/32', device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading CLIP model: {model_name} on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        print("CLIP model loaded!")
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """提取单张图片的CLIP特征"""
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(image_input)
            features = features.cpu().numpy().squeeze()
        
        return features.astype(np.float32)
    
    def extract_batch_features(self, image_dir: Path, output_dir: Path) -> dict:
        """批量提取图片特征并保存"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        print(f"Found {len(image_files)} images in {image_dir}")
        
        features_dict = {}
        
        for img_path in tqdm(image_files, desc="Extracting features"):
            try:
                features = self.extract_features(str(img_path))
                item_id = img_path.stem
                np.save(output_dir / f"{item_id}.npy", features)
                features_dict[item_id] = features
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
        
        return features_dict


def extract_all_features(base_dir: Path, split: str = 'toys'):
    """提取原始图片和攻击图片的CLIP特征"""
    
    extractor = CLIPFeatureExtractor()
    
    # 原始图片目录
    original_dir = base_dir / split
    original_features_dir = base_dir / 'features' / 'vitb32_features' / f'{split}_original'
    
    # 攻击图片目录  
    attacked_dir = base_dir / f'{split}2'
    attacked_features_dir = base_dir / 'features' / 'vitb32_features' / f'{split}_attacked'
    
    print("\n" + "="*60)
    print("Step 1: Extracting features from ORIGINAL images")
    print("="*60)
    original_features = extractor.extract_batch_features(original_dir, original_features_dir)
    
    print("\n" + "="*60)
    print("Step 2: Extracting features from ATTACKED images")
    print("="*60)
    attacked_features = extractor.extract_batch_features(attacked_dir, attacked_features_dir)
    
    return original_features, attacked_features


def evaluate_attack_effect(base_dir: Path, split: str = 'toys', num_samples: int = None):
    """
    评估攻击效果
    
    比较指标：
    1. 特征距离：原始特征和攻击特征的L2距离
    2. 余弦相似度：原始特征和攻击特征的相似度
    3. 特征变化程度
    """
    
    print("\n" + "="*60)
    print("VIP5 Typography Attack Evaluation")
    print("="*60)
    
    # 特征目录
    original_features_dir = base_dir / 'features' / 'vitb32_features' / f'{split}_original'
    attacked_features_dir = base_dir / 'features' / 'vitb32_features' / f'{split}_attacked'
    
    # 检查特征是否已提取
    if not original_features_dir.exists() or not attacked_features_dir.exists():
        print("Features not found. Extracting features first...")
        extract_all_features(base_dir, split)
    
    # 加载特征
    print("\nLoading features...")
    original_features = {}
    attacked_features = {}
    
    for f in original_features_dir.glob('*.npy'):
        original_features[f.stem] = np.load(f)
    
    for f in attacked_features_dir.glob('*.npy'):
        attacked_features[f.stem] = np.load(f)
    
    print(f"Original features: {len(original_features)}")
    print(f"Attacked features: {len(attacked_features)}")
    
    # 找到共同的物品
    common_items = set(original_features.keys()) & set(attacked_features.keys())
    print(f"Common items: {len(common_items)}")
    
    if len(common_items) == 0:
        print("Error: No common items found!")
        return
    
    # 评估
    results = {
        'feature_distances': [],
        'cosine_similarities': [],
        'detailed_results': [],
    }
    
    common_items_list = list(common_items)
    
    if num_samples and num_samples < len(common_items_list):
        import random
        random.seed(42)
        sample_items = random.sample(common_items_list, num_samples)
    else:
        sample_items = common_items_list
    
    print(f"\nEvaluating {len(sample_items)} items...")
    
    for item_id in tqdm(sample_items, desc="Evaluating"):
        orig_feat = original_features[item_id]
        attack_feat = attacked_features[item_id]
        
        # 计算特征距离
        l2_dist = np.linalg.norm(attack_feat - orig_feat)
        cosine_sim = np.dot(orig_feat, attack_feat) / (
            np.linalg.norm(orig_feat) * np.linalg.norm(attack_feat) + 1e-8
        )
        
        results['feature_distances'].append(l2_dist)
        results['cosine_similarities'].append(cosine_sim)
        
        results['detailed_results'].append({
            'item_id': item_id,
            'l2_distance': float(l2_dist),
            'cosine_similarity': float(cosine_sim),
        })
    
    # 计算统计
    results['summary'] = {
        'total_items': len(sample_items),
        'avg_l2_distance': float(np.mean(results['feature_distances'])),
        'std_l2_distance': float(np.std(results['feature_distances'])),
        'avg_cosine_similarity': float(np.mean(results['cosine_similarities'])),
        'std_cosine_similarity': float(np.std(results['cosine_similarities'])),
        'min_cosine_similarity': float(np.min(results['cosine_similarities'])),
        'max_cosine_similarity': float(np.max(results['cosine_similarities'])),
    }
    
    # 打印结果
    print("\n" + "="*60)
    print("Attack Evaluation Results")
    print("="*60)
    print(f"\nFeature Space Analysis:")
    print(f"  Total items: {results['summary']['total_items']}")
    print(f"  Avg L2 distance: {results['summary']['avg_l2_distance']:.4f} ± {results['summary']['std_l2_distance']:.4f}")
    print(f"  Avg cosine similarity: {results['summary']['avg_cosine_similarity']:.4f} ± {results['summary']['std_cosine_similarity']:.4f}")
    print(f"  Cosine range: [{results['summary']['min_cosine_similarity']:.4f}, {results['summary']['max_cosine_similarity']:.4f}]")
    
    # 保存结果
    output_dir = base_dir / 'attack_evaluation'
    output_dir.mkdir(exist_ok=True)
    
    summary_path = output_dir / 'evaluation_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results['summary'], f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


def compare_recommendations(base_dir: Path, split: str = 'toys',
                           top_k: int = 10, num_queries: int = 100):
    """
    比较原始特征和攻击特征的推荐结果差异
    
    模拟场景：
    - 随机选择物品作为"被攻击的目标物品"
    - 随机选择物品作为"查询物品"（模拟用户历史）
    - 比较目标物品在推荐列表中的排名变化
    """
    
    print("\n" + "="*60)
    print("Recommendation Comparison Analysis")
    print("="*60)
    
    # 加载特征
    original_features_dir = base_dir / 'features' / 'vitb32_features' / f'{split}_original'
    attacked_features_dir = base_dir / 'features' / 'vitb32_features' / f'{split}_attacked'
    
    if not original_features_dir.exists() or not attacked_features_dir.exists():
        print("Features not found. Please run with --mode extract first")
        return
    
    original_features = {}
    attacked_features = {}
    
    for f in original_features_dir.glob('*.npy'):
        original_features[f.stem] = np.load(f)
    
    for f in attacked_features_dir.glob('*.npy'):
        attacked_features[f.stem] = np.load(f)
    
    common_items = list(set(original_features.keys()) & set(attacked_features.keys()))
    print(f"Common items: {len(common_items)}")
    
    if len(common_items) < 10:
        print("Not enough items for comparison")
        return
    
    import random
    random.seed(42)
    
    # 选择目标物品
    num_targets = min(num_queries, len(common_items) // 2)
    target_items = random.sample(common_items, num_targets)
    
    results = {
        'comparisons': [],
        'rank_improvements': [],
    }
    
    print(f"\nComparing {len(target_items)} target items with Top-{top_k}...")
    
    for target_id in tqdm(target_items, desc="Comparing"):
        # 随机选择查询物品
        query_items = random.sample([x for x in common_items if x != target_id], 
                                    min(5, len(common_items) - 1))
        
        # 计算查询特征均值
        query_features = np.mean([original_features[q] for q in query_items], axis=0)
        
        # 候选集
        candidate_items = [x for x in common_items if x not in query_items]
        
        # 原始特征排名
        orig_scores = []
        for item_id in candidate_items:
            feat = original_features[item_id]
            sim = np.dot(query_features, feat) / (
                np.linalg.norm(query_features) * np.linalg.norm(feat) + 1e-8
            )
            orig_scores.append((item_id, sim))
        orig_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 攻击特征排名（只有目标物品用攻击特征）
        attack_scores = []
        for item_id in candidate_items:
            if item_id == target_id:
                feat = attacked_features[item_id]
            else:
                feat = original_features[item_id]
            sim = np.dot(query_features, feat) / (
                np.linalg.norm(query_features) * np.linalg.norm(feat) + 1e-8
            )
            attack_scores.append((item_id, sim))
        attack_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 计算排名
        orig_rank = next((i+1 for i, (id, _) in enumerate(orig_scores) if id == target_id), -1)
        attack_rank = next((i+1 for i, (id, _) in enumerate(attack_scores) if id == target_id), -1)
        
        rank_change = orig_rank - attack_rank
        
        results['comparisons'].append({
            'target_id': target_id,
            'original_rank': orig_rank,
            'attack_rank': attack_rank,
            'rank_change': rank_change,
            'in_orig_top_k': orig_rank <= top_k,
            'in_attack_top_k': attack_rank <= top_k,
            'entered_top_k': attack_rank <= top_k and orig_rank > top_k,
        })
        
        results['rank_improvements'].append(rank_change)
    
    # 统计
    improvements = results['rank_improvements']
    entered_top_k = sum(1 for c in results['comparisons'] if c['entered_top_k'])
    
    results['summary'] = {
        'total_queries': len(target_items),
        'top_k': top_k,
        'avg_rank_change': float(np.mean(improvements)),
        'std_rank_change': float(np.std(improvements)),
        'median_rank_change': float(np.median(improvements)),
        'max_improvement': int(np.max(improvements)),
        'max_degradation': int(np.min(improvements)),
        'improved_count': sum(1 for x in improvements if x > 0),
        'degraded_count': sum(1 for x in improvements if x < 0),
        'unchanged_count': sum(1 for x in improvements if x == 0),
        'entered_top_k': entered_top_k,
        'success_rate': sum(1 for x in improvements if x > 0) / len(improvements),
    }
    
    # 打印结果
    print("\n" + "="*60)
    print("Recommendation Comparison Results")
    print("="*60)
    print(f"\nSettings:")
    print(f"  Top-K: {top_k}")
    print(f"  Total queries: {results['summary']['total_queries']}")
    
    print(f"\nRank Changes (positive = attack helped):")
    print(f"  Average: {results['summary']['avg_rank_change']:.2f} ± {results['summary']['std_rank_change']:.2f}")
    print(f"  Median: {results['summary']['median_rank_change']:.1f}")
    print(f"  Best improvement: +{results['summary']['max_improvement']} positions")
    print(f"  Worst degradation: {results['summary']['max_degradation']} positions")
    
    print(f"\nAttack Success Rate:")
    print(f"  Rank improved: {results['summary']['improved_count']} ({results['summary']['success_rate']*100:.1f}%)")
    print(f"  Rank degraded: {results['summary']['degraded_count']} ({results['summary']['degraded_count']/len(improvements)*100:.1f}%)")
    print(f"  Unchanged: {results['summary']['unchanged_count']}")
    print(f"  Newly entered Top-{top_k}: {results['summary']['entered_top_k']}")
    
    # 保存结果
    output_dir = base_dir / 'attack_evaluation'
    output_dir.mkdir(exist_ok=True)
    
    comparison_path = output_dir / 'recommendation_comparison.json'
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': results['summary'],
            'comparisons': results['comparisons']
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {comparison_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate VIP5 Typography Attack Effect',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: extract features + evaluate + compare
  python evaluate_attack.py --mode full --split toys
  
  # Only extract features
  python evaluate_attack.py --mode extract --split toys
  
  # Only evaluate features
  python evaluate_attack.py --mode evaluate --split toys
  
  # Compare recommendations
  python evaluate_attack.py --mode compare --split toys --top_k 10
        """
    )
    
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'extract', 'evaluate', 'compare'],
                        help='Evaluation mode')
    parser.add_argument('--split', type=str, default='toys',
                        help='Dataset split')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='VIP5 base directory')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Top-K for recommendation comparison')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate')
    parser.add_argument('--num_queries', type=int, default=100,
                        help='Number of queries for comparison')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir).resolve() if args.base_dir else SCRIPT_DIR
    
    if args.mode == 'full':
        extract_all_features(base_dir, args.split)
        evaluate_attack_effect(base_dir, args.split, num_samples=args.num_samples)
        compare_recommendations(base_dir, args.split, top_k=args.top_k, num_queries=args.num_queries)
        
    elif args.mode == 'extract':
        extract_all_features(base_dir, args.split)
        
    elif args.mode == 'evaluate':
        evaluate_attack_effect(base_dir, args.split, num_samples=args.num_samples)
        
    elif args.mode == 'compare':
        compare_recommendations(base_dir, args.split, top_k=args.top_k, num_queries=args.num_queries)


if __name__ == '__main__':
    main()
