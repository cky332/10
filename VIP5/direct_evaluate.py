#!/usr/bin/env python3
"""
VIP5 Direct Feature Evaluation
==============================
直接使用特征文件进行评估，完全绕过ID映射问题

核心思路：
- 不依赖 item2img_dict 或 id2item 映射
- 直接使用特征文件名作为商品标识符
- 通过CLIP特征相似度来评估攻击效果
- 同时提供VIP5模型评估（使用特征文件名作为伪ID）

使用方法：
    # 基于CLIP特征的评估（推荐，不需要VIP5模型）
    python direct_evaluate.py --mode clip --split toys

    # 基于VIP5模型的评估
    python direct_evaluate.py --mode vip5 --split toys --num_samples 100
"""

import os
import sys
import re
import argparse
import pickle
import json
import numpy as np
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / 'src'))


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


# ============================================================
# CLIP特征评估（推荐方法）
# ============================================================

def evaluate_with_clip_features(split='toys', num_samples=100, num_candidates=20):
    """
    使用CLIP特征直接评估攻击效果

    这是最可靠的评估方法，因为：
    1. 不依赖任何ID映射
    2. 直接比较特征相似度
    3. 模拟推荐系统的核心逻辑
    """

    print("\n" + "="*70)
    print("CLIP Feature-based Attack Evaluation")
    print("="*70)

    # 加载特征
    original_dir = SCRIPT_DIR / 'features' / 'vitb32_features' / f'{split}_original'
    attacked_dir = SCRIPT_DIR / 'features' / 'vitb32_features' / f'{split}_attacked'

    if not original_dir.exists() or not attacked_dir.exists():
        print("Error: Feature directories not found!")
        print(f"  Original: {original_dir}")
        print(f"  Attacked: {attacked_dir}")
        return None

    # 加载所有特征
    print("\nLoading features...")
    original_features = {}
    attacked_features = {}

    for f in original_dir.glob('*.npy'):
        original_features[f.stem] = np.load(f)

    for f in attacked_dir.glob('*.npy'):
        attacked_features[f.stem] = np.load(f)

    # 找到共同的商品（被攻击的商品）
    common_items = list(set(original_features.keys()) & set(attacked_features.keys()))

    print(f"Original features: {len(original_features)}")
    print(f"Attacked features: {len(attacked_features)}")
    print(f"Common items (attacked): {len(common_items)}")

    if len(common_items) < 10:
        print("Error: Not enough common items!")
        return None

    # 所有可用的商品（用于构建候选集）
    all_items = list(original_features.keys())

    print(f"\n{'='*70}")
    print(f"Evaluating {min(num_samples, len(common_items))} attacked items")
    print(f"Each with {num_candidates} candidates")
    print(f"{'='*70}")

    random.seed(42)
    np.random.seed(42)

    # 选择要评估的目标商品
    target_items = random.sample(common_items, min(num_samples, len(common_items)))

    results = []

    for i, target_item in enumerate(tqdm(target_items, desc="Evaluating")):
        # 获取目标商品的原始和攻击特征
        orig_target_feat = original_features[target_item]
        attack_target_feat = attacked_features[target_item]

        # 构建候选集（目标 + 负样本）
        negative_pool = [item for item in all_items if item != target_item]
        if len(negative_pool) < num_candidates - 1:
            continue

        negative_items = random.sample(negative_pool, num_candidates - 1)
        candidate_items = negative_items + [target_item]
        random.shuffle(candidate_items)

        # 模拟查询：使用随机商品的特征作为"用户偏好"
        query_items = random.sample([x for x in all_items if x not in candidate_items],
                                    min(5, len(all_items) - num_candidates))
        query_feat = np.mean([original_features[q] for q in query_items], axis=0)
        query_feat = query_feat / (np.linalg.norm(query_feat) + 1e-8)

        # 计算原始排名
        orig_scores = []
        for item in candidate_items:
            feat = original_features[item]
            feat = feat / (np.linalg.norm(feat) + 1e-8)
            score = np.dot(query_feat, feat)
            orig_scores.append((item, score))

        orig_scores.sort(key=lambda x: x[1], reverse=True)
        orig_rank = next((i+1 for i, (item, _) in enumerate(orig_scores) if item == target_item), -1)

        # 计算攻击后排名（只有目标商品使用攻击特征）
        attack_scores = []
        for item in candidate_items:
            if item == target_item:
                feat = attack_target_feat
            else:
                feat = original_features[item]
            feat = feat / (np.linalg.norm(feat) + 1e-8)
            score = np.dot(query_feat, feat)
            attack_scores.append((item, score))

        attack_scores.sort(key=lambda x: x[1], reverse=True)
        attack_rank = next((i+1 for i, (item, _) in enumerate(attack_scores) if item == target_item), -1)

        # 计算排名变化
        rank_change = orig_rank - attack_rank  # 正数 = 攻击有效

        # 计算特征变化
        feat_diff = np.linalg.norm(attack_target_feat - orig_target_feat)
        cosine_sim = np.dot(orig_target_feat, attack_target_feat) / (
            np.linalg.norm(orig_target_feat) * np.linalg.norm(attack_target_feat) + 1e-8
        )

        results.append({
            'target_item': target_item,
            'original_rank': orig_rank,
            'attacked_rank': attack_rank,
            'rank_change': rank_change,
            'feature_diff': float(feat_diff),
            'cosine_similarity': float(cosine_sim),
        })

        # 打印前几个样本
        if i < 5:
            print(f"\n  Sample {i+1}: {target_item}")
            print(f"  Original Rank: {orig_rank}, Attacked Rank: {attack_rank}")
            print(f"  Rank Change: {rank_change:+d}")
            print(f"  Feature Diff: {feat_diff:.4f}, Cosine Sim: {cosine_sim:.4f}")

    # 统计结果
    rank_changes = [r['rank_change'] for r in results]
    improved = sum(1 for x in rank_changes if x > 0)
    degraded = sum(1 for x in rank_changes if x < 0)
    unchanged = sum(1 for x in rank_changes if x == 0)

    feat_diffs = [r['feature_diff'] for r in results]
    cosine_sims = [r['cosine_similarity'] for r in results]

    print(f"\n{'='*70}")
    print("EVALUATION RESULTS (CLIP Feature-based)")
    print(f"{'='*70}")

    print(f"\n[Settings]")
    print(f"  Total samples: {len(results)}")
    print(f"  Candidates per sample: {num_candidates}")

    print(f"\n[Rank Changes] (positive = attack helped)")
    print(f"  Mean:   {np.mean(rank_changes):+.2f}")
    print(f"  Median: {np.median(rank_changes):+.2f}")
    print(f"  Std:    {np.std(rank_changes):.2f}")
    print(f"  Best:   {np.max(rank_changes):+d}")
    print(f"  Worst:  {np.min(rank_changes):+d}")

    print(f"\n[Attack Success Rate]")
    print(f"  Improved: {improved} ({improved/len(results)*100:.1f}%)")
    print(f"  Degraded: {degraded} ({degraded/len(results)*100:.1f}%)")
    print(f"  Unchanged: {unchanged} ({unchanged/len(results)*100:.1f}%)")

    print(f"\n[Feature Change Statistics]")
    print(f"  Avg L2 Distance: {np.mean(feat_diffs):.4f}")
    print(f"  Avg Cosine Similarity: {np.mean(cosine_sims):.4f}")
    print(f"  Min Cosine Similarity: {np.min(cosine_sims):.4f}")

    # 分析
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    success_rate = improved / len(results)
    if success_rate > 0.5:
        print(f"\n✓ Attack is EFFECTIVE!")
        print(f"  Success rate: {success_rate*100:.1f}%")
    elif success_rate > 0.3:
        print(f"\n⚠ Attack has MODERATE effect")
        print(f"  Success rate: {success_rate*100:.1f}%")
    else:
        print(f"\n✗ Attack has LIMITED effect")
        print(f"  Success rate: {success_rate*100:.1f}%")

        if np.mean(feat_diffs) < 1.0:
            print("\n  Suggestion: Feature change is small. Try:")
            print("    - Increase epsilon (e.g., --epsilon 0.15)")
            print("    - Increase iterations (e.g., --max_iter 150)")
        else:
            print("\n  Feature change is significant but doesn't improve ranking.")
            print("  Suggestion: Try different attack strategies:")
            print("    - Use --attack_type toward_target")
            print("    - Use --attack_type multi_direction")

    # 保存结果
    output_dir = SCRIPT_DIR / 'attack_evaluation_direct'
    output_dir.mkdir(exist_ok=True)

    summary = {
        'method': 'clip_feature_based',
        'total_samples': len(results),
        'num_candidates': num_candidates,
        'rank_change_mean': float(np.mean(rank_changes)),
        'rank_change_median': float(np.median(rank_changes)),
        'success_rate': float(success_rate),
        'improved': improved,
        'degraded': degraded,
        'unchanged': unchanged,
        'avg_feature_diff': float(np.mean(feat_diffs)),
        'avg_cosine_sim': float(np.mean(cosine_sims)),
    }

    with open(output_dir / 'clip_evaluation_results.json', 'w') as f:
        json.dump({
            'summary': summary,
            'detailed_results': results
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return summary, results


# ============================================================
# VIP5模型评估（使用伪ID）
# ============================================================

def evaluate_with_vip5(split='toys', num_samples=100, num_candidates=20, checkpoint=None):
    """
    使用VIP5模型评估，但使用特征文件名作为伪商品ID

    注意：这种方法可能不如CLIP特征评估准确，因为：
    1. VIP5模型可能依赖于特定的ID embedding
    2. 使用非训练时的ID可能影响模型行为
    """

    from transformers import T5Config
    from tokenization import P5Tokenizer
    from model import VIP5Tuning
    from utils import load_state_dict
    from adapters import AdapterConfig

    print("\n" + "="*70)
    print("VIP5 Model-based Attack Evaluation (Pseudo-ID Mode)")
    print("="*70)

    # 创建参数
    args = create_vip5_args(split, checkpoint)
    config = create_vip5_config(args)

    # 加载特征
    original_dir = SCRIPT_DIR / 'features' / 'vitb32_features' / f'{split}_original'
    attacked_dir = SCRIPT_DIR / 'features' / 'vitb32_features' / f'{split}_attacked'

    print("\nLoading features...")
    original_features = {}
    attacked_features = {}

    for f in original_dir.glob('*.npy'):
        original_features[f.stem] = np.load(f)

    for f in attacked_dir.glob('*.npy'):
        attacked_features[f.stem] = np.load(f)

    common_items = list(set(original_features.keys()) & set(attacked_features.keys()))
    all_items = list(original_features.keys())

    print(f"Original features: {len(original_features)}")
    print(f"Attacked features: {len(attacked_features)}")
    print(f"Common items: {len(common_items)}")

    if len(common_items) < 10:
        print("Error: Not enough common items!")
        return None

    # 加载VIP5模型
    print("\nLoading VIP5 model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = P5Tokenizer.from_pretrained(
        args.backbone,
        max_length=args.max_text_length,
        do_lower_case=args.do_lower_case
    )

    model = VIP5Tuning.from_pretrained(args.backbone, config=config)
    model.to(device)
    model.resize_token_embeddings(tokenizer.vocab_size)
    model.tokenizer = tokenizer

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        state_dict = load_state_dict(args.checkpoint, 'cpu')
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    print("VIP5 model ready!")

    # 评估
    random.seed(42)
    np.random.seed(42)

    target_items = random.sample(common_items, min(num_samples, len(common_items)))

    print(f"\n{'='*70}")
    print(f"Evaluating {len(target_items)} samples")
    print(f"{'='*70}")

    results = []

    for i, target_item in enumerate(tqdm(target_items, desc="Evaluating")):
        # 构建候选集
        negative_pool = [item for item in all_items if item != target_item]
        if len(negative_pool) < num_candidates - 1:
            continue

        negative_items = random.sample(negative_pool, num_candidates - 1)
        candidate_items = negative_items + [target_item]
        random.shuffle(candidate_items)

        # 为每个候选商品分配一个数字ID（用于模型输入）
        item_to_pseudo_id = {item: str(idx+1) for idx, item in enumerate(candidate_items)}
        pseudo_id_to_item = {v: k for k, v in item_to_pseudo_id.items()}

        target_pseudo_id = item_to_pseudo_id[target_item]

        # 构造输入
        candidates_text = ' {}, '.format('<extra_id_0> ' * args.image_feature_size_ratio).join(
            [item_to_pseudo_id[c] for c in candidate_items]
        ) + ' <extra_id_0>' * args.image_feature_size_ratio

        source_text = f"We want to make recommendation for user_1 .  Select the best item from these candidates : \n {candidates_text}"

        # 原始特征评估
        orig_feats = np.stack([original_features[item] for item in candidate_items])
        orig_recs = get_vip5_recommendations(
            model, tokenizer, source_text, orig_feats, device, args
        )
        orig_rank = get_rank(orig_recs, target_pseudo_id, num_candidates)

        # 攻击特征评估
        attack_feats = []
        for item in candidate_items:
            if item == target_item:
                attack_feats.append(attacked_features[item])
            else:
                attack_feats.append(original_features[item])
        attack_feats = np.stack(attack_feats)

        attack_recs = get_vip5_recommendations(
            model, tokenizer, source_text, attack_feats, device, args
        )
        attack_rank = get_rank(attack_recs, target_pseudo_id, num_candidates)

        rank_change = orig_rank - attack_rank

        results.append({
            'target_item': target_item,
            'original_rank': orig_rank,
            'attacked_rank': attack_rank,
            'rank_change': rank_change,
        })

        if i < 5:
            print(f"\n  Sample {i+1}: {target_item}")
            print(f"  Original Rank: {orig_rank}, Attacked Rank: {attack_rank}")
            print(f"  Rank Change: {rank_change:+d}")

    # 统计
    rank_changes = [r['rank_change'] for r in results]
    improved = sum(1 for x in rank_changes if x > 0)
    degraded = sum(1 for x in rank_changes if x < 0)
    unchanged = sum(1 for x in rank_changes if x == 0)

    print(f"\n{'='*70}")
    print("EVALUATION RESULTS (VIP5 Model-based)")
    print(f"{'='*70}")

    print(f"\n[Settings]")
    print(f"  Total samples: {len(results)}")
    print(f"  Candidates per sample: {num_candidates}")

    print(f"\n[Rank Changes]")
    print(f"  Mean:   {np.mean(rank_changes):+.2f}")
    print(f"  Median: {np.median(rank_changes):+.2f}")
    print(f"  Best:   {np.max(rank_changes):+d}")
    print(f"  Worst:  {np.min(rank_changes):+d}")

    print(f"\n[Attack Success Rate]")
    print(f"  Improved: {improved} ({improved/len(results)*100:.1f}%)")
    print(f"  Degraded: {degraded} ({degraded/len(results)*100:.1f}%)")
    print(f"  Unchanged: {unchanged} ({unchanged/len(results)*100:.1f}%)")

    # 保存结果
    output_dir = SCRIPT_DIR / 'attack_evaluation_direct'
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'vip5_evaluation_results.json', 'w') as f:
        json.dump({
            'summary': {
                'total_samples': len(results),
                'success_rate': improved / len(results),
                'mean_rank_change': float(np.mean(rank_changes)),
            },
            'detailed_results': results
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return results


def create_vip5_args(split, checkpoint=None):
    """创建VIP5参数"""
    class Args:
        pass

    args = Args()
    args.split = split
    args.backbone = '/home/mlsnrs/data/cky/5-main/t5-small-local'
    args.image_feature_type = 'vitb32'
    args.image_feature_size_ratio = 2
    args.image_feature_dim = 512
    args.use_adapter = True
    args.reduction_factor = 8
    args.use_single_adapter = True
    args.use_vis_layer_norm = True
    args.add_adapter_cross_attn = True
    args.use_lm_head_adapter = False
    args.dropout = 0.1
    args.losses = 'sequential,direct,explanation'
    args.max_text_length = 1024
    args.do_lower_case = False

    if checkpoint:
        args.checkpoint = checkpoint
    else:
        args.checkpoint = str(SCRIPT_DIR / 'snap' / 'toys-vitb32-2-8-20' / 'BEST_EVAL_LOSS.pth')

    return args


def create_vip5_config(args):
    """创建VIP5配置"""
    from transformers import T5Config
    from adapters import AdapterConfig

    config = T5Config.from_pretrained(args.backbone)

    config.feat_dim = 512
    config.n_vis_tokens = args.image_feature_size_ratio
    config.use_vis_layer_norm = args.use_vis_layer_norm
    config.reduction_factor = args.reduction_factor
    config.use_adapter = args.use_adapter
    config.add_adapter_cross_attn = args.add_adapter_cross_attn
    config.use_lm_head_adapter = args.use_lm_head_adapter
    config.use_single_adapter = args.use_single_adapter
    config.dropout_rate = args.dropout
    config.dropout = args.dropout
    config.attention_dropout = args.dropout
    config.activation_dropout = args.dropout
    config.losses = args.losses

    tasks = re.split("[, ]+", args.losses)
    config.adapter_config = AdapterConfig()
    config.adapter_config.tasks = tasks
    config.adapter_config.d_model = config.d_model
    config.adapter_config.use_single_adapter = args.use_single_adapter
    config.adapter_config.reduction_factor = args.reduction_factor
    config.adapter_config.track_z = False

    return config


def get_vip5_recommendations(model, tokenizer, source_text, vis_feats, device, args):
    """获取VIP5推荐结果"""

    input_ids = tokenizer.encode(
        source_text, padding=True, truncation=True,
        max_length=args.max_text_length
    )
    tokenized_text = tokenizer.tokenize(source_text)

    # whole word ids
    whole_word_ids = []
    curr = 0
    for i in range(len(tokenized_text)):
        if tokenized_text[i].startswith('▁') or tokenized_text[i] == '<extra_id_0>':
            curr += 1
        whole_word_ids.append(curr)
    whole_word_ids = whole_word_ids[:len(input_ids) - 1] + [0]

    category_ids = [1 if token_id == 32099 else 0 for token_id in input_ids]

    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    whole_word_ids = torch.LongTensor(whole_word_ids).unsqueeze(0).to(device)
    category_ids = torch.LongTensor(category_ids).unsqueeze(0).to(device)
    vis_feats = torch.from_numpy(vis_feats).float().unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            whole_word_ids=whole_word_ids,
            category_ids=category_ids,
            vis_feats=vis_feats,
            task='direct',
            max_length=50,
            num_beams=20,
            no_repeat_ngram_size=0,
            num_return_sequences=20,
            early_stopping=True
        )

    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    recommendations = []
    for text in generated:
        item_id = text.strip()
        if item_id and item_id not in recommendations:
            recommendations.append(item_id)

    return recommendations


def get_rank(recommendations, target_id, num_candidates):
    """获取目标在推荐列表中的排名"""
    try:
        return recommendations.index(target_id) + 1
    except ValueError:
        return num_candidates + 1


def main():
    parser = argparse.ArgumentParser(
        description='VIP5 Direct Feature Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基于CLIP特征评估（推荐，快速且可靠）
  python direct_evaluate.py --mode clip --split toys

  # 基于VIP5模型评估
  python direct_evaluate.py --mode vip5 --split toys --num_samples 100

  # 更多候选商品
  python direct_evaluate.py --mode clip --split toys --num_candidates 50
        """
    )

    parser.add_argument('--mode', type=str, default='clip',
                        choices=['clip', 'vip5'],
                        help='评估模式: clip (特征相似度) 或 vip5 (模型推理)')
    parser.add_argument('--split', type=str, default='toys',
                        help='数据集名称')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='评估样本数量')
    parser.add_argument('--num_candidates', type=int, default=20,
                        help='每个样本的候选商品数量')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='VIP5模型检查点路径')

    args = parser.parse_args()

    if args.mode == 'clip':
        evaluate_with_clip_features(
            split=args.split,
            num_samples=args.num_samples,
            num_candidates=args.num_candidates
        )
    elif args.mode == 'vip5':
        evaluate_with_vip5(
            split=args.split,
            num_samples=args.num_samples,
            num_candidates=args.num_candidates,
            checkpoint=args.checkpoint
        )


if __name__ == '__main__':
    main()
