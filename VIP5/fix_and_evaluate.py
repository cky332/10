#!/usr/bin/env python3
"""
VIP5 Attack Evaluation - Fixed Version
=======================================
修复版评估脚本 - 解决文件名映射不匹配问题

问题诊断：
- transfer_attack.py 生成的攻击图片文件名与 item2img_dict 映射不一致
- 导致 evaluate_attack_vip5.py 只能匹配到很少的样本

解决方案：
1. 自动检测文件名格式并建立正确的映射
2. 直接使用特征文件名作为商品标识符
3. 支持多种文件名格式 (ASIN, 数字ID, 等)

使用方法：
    # 诊断问题
    python fix_and_evaluate.py --mode diagnose --split toys

    # 修复映射并评估
    python fix_and_evaluate.py --mode evaluate --split toys --num_samples 100

作者: 对抗攻击研究
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
from collections import defaultdict, Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.backends.cudnn as cudnn

# 设置路径
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / 'src'))

from transformers import T5Config
from tokenization import P5Tokenizer
from model import VIP5Tuning
from utils import load_state_dict


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def ReadLineFromFile(path):
    lines = []
    with open(path, 'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines


# ============================================================
# 诊断功能
# ============================================================

def diagnose_mapping(split='toys'):
    """诊断文件名映射问题"""

    print("\n" + "="*70)
    print("DIAGNOSIS: File Name Mapping Analysis")
    print("="*70)

    data_dir = SCRIPT_DIR / 'data' / split
    original_feat_dir = SCRIPT_DIR / 'features' / 'vitb32_features' / f'{split}_original'
    attacked_feat_dir = SCRIPT_DIR / 'features' / 'vitb32_features' / f'{split}_attacked'
    image_dir = SCRIPT_DIR / split
    attacked_image_dir = SCRIPT_DIR / f'{split}2'

    print(f"\n[1] Checking directories...")
    print(f"  Data dir: {data_dir} - {'EXISTS' if data_dir.exists() else 'NOT FOUND'}")
    print(f"  Image dir: {image_dir} - {'EXISTS' if image_dir.exists() else 'NOT FOUND'}")
    print(f"  Attacked image dir: {attacked_image_dir} - {'EXISTS' if attacked_image_dir.exists() else 'NOT FOUND'}")
    print(f"  Original features: {original_feat_dir} - {'EXISTS' if original_feat_dir.exists() else 'NOT FOUND'}")
    print(f"  Attacked features: {attacked_feat_dir} - {'EXISTS' if attacked_feat_dir.exists() else 'NOT FOUND'}")

    # 加载数据映射
    print(f"\n[2] Loading data mappings...")
    datamaps_path = data_dir / 'datamaps.json'
    item2img_path = data_dir / 'item2img_dict.pkl'

    if datamaps_path.exists():
        datamaps = load_json(str(datamaps_path))
        id2item = datamaps.get('id2item', {})
        item2id = datamaps.get('item2id', {})
        print(f"  datamaps.json: {len(id2item)} items (id2item)")
        print(f"  Sample id2item: {list(id2item.items())[:3]}")
    else:
        print(f"  datamaps.json: NOT FOUND")
        id2item = {}
        item2id = {}

    if item2img_path.exists():
        item2img_dict = load_pickle(str(item2img_path))
        print(f"  item2img_dict.pkl: {len(item2img_dict)} items")
        sample_items = list(item2img_dict.items())[:3]
        print(f"  Sample item2img: {sample_items}")
    else:
        print(f"  item2img_dict.pkl: NOT FOUND")
        item2img_dict = {}

    # 检查特征文件
    print(f"\n[3] Analyzing feature files...")

    if original_feat_dir.exists():
        original_files = set(f.stem for f in original_feat_dir.glob('*.npy'))
        print(f"  Original feature files: {len(original_files)}")
        print(f"  Sample names: {list(original_files)[:5]}")
    else:
        original_files = set()
        print(f"  Original feature files: 0 (directory not found)")

    if attacked_feat_dir.exists():
        attacked_files = set(f.stem for f in attacked_feat_dir.glob('*.npy'))
        print(f"  Attacked feature files: {len(attacked_files)}")
        print(f"  Sample names: {list(attacked_files)[:5]}")
    else:
        attacked_files = set()
        print(f"  Attacked feature files: 0 (directory not found)")

    common_files = original_files & attacked_files
    print(f"  Common files: {len(common_files)}")

    # 分析映射问题
    print(f"\n[4] Analyzing mapping issues...")

    # 检查特征文件名是否可以直接匹配到商品ID
    matched_via_id2item = 0
    matched_via_item2img = 0
    matched_directly = 0

    for feat_name in list(common_files)[:100]:  # 检查前100个
        # 方式1: 特征文件名是否在id2item的values中 (ASIN)
        if feat_name in item2id:
            matched_directly += 1

        # 方式2: 通过item2img_dict
        for asin, img_info in item2img_dict.items():
            if isinstance(img_info, str):
                img_name = img_info.rsplit('.', 1)[0] if '.' in img_info else img_info
                if '/' in img_name:
                    img_name = img_name.split('/')[-1]
            elif isinstance(img_info, list) and len(img_info) > 0:
                img_name = img_info[0].rsplit('.', 1)[0] if '.' in img_info[0] else img_info[0]
            else:
                continue

            if img_name == feat_name:
                matched_via_item2img += 1
                break

    print(f"  Feature names matching ASIN directly: {matched_directly}/100")
    print(f"  Feature names matching via item2img: {matched_via_item2img}/100")

    # 判断文件名格式
    print(f"\n[5] Detecting file name format...")
    sample_feat_names = list(common_files)[:10]

    asin_pattern = 0
    numeric_pattern = 0
    other_pattern = 0

    for name in sample_feat_names:
        if re.match(r'^B[0-9A-Z]{9}$', name):  # ASIN格式
            asin_pattern += 1
        elif name.isdigit():  # 纯数字
            numeric_pattern += 1
        else:
            other_pattern += 1

    print(f"  ASIN format (B0xxxxxxxxx): {asin_pattern}")
    print(f"  Numeric format: {numeric_pattern}")
    print(f"  Other format: {other_pattern}")

    # 建议
    print(f"\n[6] DIAGNOSIS RESULT:")
    print("="*70)

    if len(common_files) == 0:
        print("PROBLEM: No common feature files found!")
        print("SOLUTION: Run attack and feature extraction first:")
        print("  python transfer_attack.py --split toys --num_images 100")
        print("  python evaluate_attack.py --mode extract --split toys")
    elif matched_directly > 50 or matched_via_item2img > 50:
        print("STATUS: Mapping looks OK, but may have partial issues")
        print("SOLUTION: Use fixed evaluation script")
    else:
        print("PROBLEM: Feature file names don't match item mapping!")
        print(f"  - Feature files named like: {sample_feat_names[:3]}")
        print(f"  - But item2img expects different format")
        print("\nSOLUTION: The fix_and_evaluate.py will handle this automatically")
        print("  Run: python fix_and_evaluate.py --mode evaluate --split toys")

    return {
        'common_files': len(common_files),
        'matched_directly': matched_directly,
        'matched_via_item2img': matched_via_item2img,
    }


# ============================================================
# 修复版评估
# ============================================================

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


def create_args(split='toys', checkpoint_path=None):
    """创建VIP5模型所需的参数"""
    args = DotDict()

    args.distributed = False
    args.multiGPU = False
    args.fp16 = True
    args.split = split
    args.train = split
    args.valid = split
    args.test = split
    args.batch_size = 1
    args.optim = 'adamw'
    args.warmup_ratio = 0.1
    args.lr = 1e-3
    args.num_workers = 0
    args.clip_grad_norm = 5.0
    args.losses = 'sequential,direct,explanation'
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

    args.epoch = 20
    args.local_rank = 0
    args.dropout = 0.1
    args.tokenizer = 'p5'
    args.max_text_length = 1024
    args.gen_max_length = 64
    args.do_lower_case = False
    args.weight_decay = 0.01
    args.adam_eps = 1e-6
    args.gradient_accumulation_steps = 1
    args.seed = 2022
    args.whole_word_embed = True
    args.category_embed = True

    args.LOSSES_NAME = ['sequential_loss', 'direct_loss', 'explanation_loss', 'total_loss']
    args.gpu = 0
    args.rank = 0

    if checkpoint_path:
        args.checkpoint = checkpoint_path
    else:
        args.checkpoint = str(SCRIPT_DIR / 'snap' / 'toys-vitb32-2-8-20' / 'BEST_EVAL_LOSS.pth')

    return args


def create_config(args):
    """创建VIP5模型配置"""
    from adapters import AdapterConfig

    config = T5Config.from_pretrained(args.backbone)

    for k, v in vars(args).items():
        setattr(config, k, v)

    image_feature_dim_dict = {
        'vitb32': 512, 'vitb16': 512, 'vitl14': 768, 'rn50': 1024, 'rn101': 512
    }
    config.feat_dim = image_feature_dim_dict[args.image_feature_type]
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

    if args.use_adapter:
        config.adapter_config = AdapterConfig()
        config.adapter_config.tasks = tasks
        config.adapter_config.d_model = config.d_model
        config.adapter_config.use_single_adapter = args.use_single_adapter
        config.adapter_config.reduction_factor = args.reduction_factor
        config.adapter_config.track_z = False
    else:
        config.adapter_config = None

    return config


class FixedEvaluationDataset:
    """
    修复版数据集 - 直接使用特征文件名作为商品标识

    关键改进：
    1. 不依赖 item2img_dict 映射
    2. 直接从特征文件名建立映射
    3. 支持多种文件名格式
    """

    def __init__(self, args, split='toys'):
        self.args = args
        self.split = split
        self.image_feature_dim = args.image_feature_dim
        self.image_feature_size_ratio = args.image_feature_size_ratio

        data_dir = SCRIPT_DIR / 'data' / split

        # 加载用户数据
        self.sequential_data = ReadLineFromFile(str(data_dir / 'sequential_data.txt'))

        # 用户-物品映射 (使用字符串ID)
        self.user_items = defaultdict(list)
        self.all_items = set()

        for line in self.sequential_data:
            parts = line.strip().split(' ')
            if len(parts) > 1:
                user = parts[0]
                items = parts[1:]
                self.user_items[user] = items
                self.all_items.update(items)

        self.user_list = list(self.user_items.keys())
        print(f"Loaded {len(self.user_list)} users, {len(self.all_items)} unique items")

        # 加载特征文件 - 核心改进
        self._load_features()

    def _load_features(self):
        """加载原始和攻击特征"""
        original_dir = SCRIPT_DIR / 'features' / 'vitb32_features' / f'{self.split}_original'
        attacked_dir = SCRIPT_DIR / 'features' / 'vitb32_features' / f'{self.split}_attacked'

        # 加载所有特征
        self.original_features = {}
        self.attacked_features = {}

        if original_dir.exists():
            for f in original_dir.glob('*.npy'):
                self.original_features[f.stem] = np.load(f)

        if attacked_dir.exists():
            for f in attacked_dir.glob('*.npy'):
                self.attacked_features[f.stem] = np.load(f)

        # 找到同时有原始和攻击特征的商品
        self.common_items = set(self.original_features.keys()) & set(self.attacked_features.keys())

        # 建立特征文件名到序列数据中商品ID的映射
        self._build_mapping()

        print(f"Original features: {len(self.original_features)}")
        print(f"Attacked features: {len(self.attacked_features)}")
        print(f"Common items with both features: {len(self.common_items)}")
        print(f"Items mappable to user sequences: {len(self.available_items)}")

    def _build_mapping(self):
        """
        建立特征文件名 <-> 用户序列中商品ID 的映射

        尝试多种匹配方式：
        1. 直接匹配
        2. 去除前缀/后缀
        3. 数字ID匹配
        """
        self.feat_to_seq_id = {}  # 特征文件名 -> 序列中的商品ID
        self.seq_id_to_feat = {}  # 序列中的商品ID -> 特征文件名

        # 方式1: 直接匹配
        for feat_name in self.common_items:
            if feat_name in self.all_items:
                self.feat_to_seq_id[feat_name] = feat_name
                self.seq_id_to_feat[feat_name] = feat_name

        # 方式2: 如果特征文件名是ASIN格式，序列中可能用数字ID
        # 尝试加载datamaps
        data_dir = SCRIPT_DIR / 'data' / self.split
        datamaps_path = data_dir / 'datamaps.json'

        if datamaps_path.exists():
            datamaps = load_json(str(datamaps_path))
            item2id = datamaps.get('item2id', {})
            id2item = datamaps.get('id2item', {})

            # ASIN -> 数字ID
            for feat_name in self.common_items:
                if feat_name not in self.feat_to_seq_id:
                    # 特征文件名可能是ASIN
                    if feat_name in item2id:
                        seq_id = str(item2id[feat_name])
                        if seq_id in self.all_items:
                            self.feat_to_seq_id[feat_name] = seq_id
                            self.seq_id_to_feat[seq_id] = feat_name

            # 数字ID -> ASIN
            for feat_name in self.common_items:
                if feat_name not in self.feat_to_seq_id:
                    # 特征文件名可能是数字ID
                    if feat_name in id2item:
                        asin = id2item[feat_name]
                        if asin in self.all_items:
                            self.feat_to_seq_id[feat_name] = asin
                            self.seq_id_to_feat[asin] = feat_name

        # 方式3: 使用item2img_dict反向查找
        item2img_path = data_dir / 'item2img_dict.pkl'
        if item2img_path.exists():
            item2img_dict = load_pickle(str(item2img_path))

            # 建立 图片文件名 -> ASIN 的反向映射
            img_to_asin = {}
            for asin, img_info in item2img_dict.items():
                if isinstance(img_info, str):
                    img_name = img_info.rsplit('.', 1)[0] if '.' in img_info else img_info
                    if '/' in img_name:
                        img_name = img_name.split('/')[-1]
                    img_to_asin[img_name] = asin
                elif isinstance(img_info, list) and len(img_info) > 0:
                    img_name = img_info[0].rsplit('.', 1)[0] if '.' in img_info[0] else img_info[0]
                    if '/' in img_name:
                        img_name = img_name.split('/')[-1]
                    img_to_asin[img_name] = asin

            for feat_name in self.common_items:
                if feat_name not in self.feat_to_seq_id:
                    if feat_name in img_to_asin:
                        asin = img_to_asin[feat_name]
                        if asin in self.all_items:
                            self.feat_to_seq_id[feat_name] = asin
                            self.seq_id_to_feat[asin] = feat_name

        self.available_items = list(self.feat_to_seq_id.keys())

        print(f"\nMapping statistics:")
        print(f"  Feature files mappable: {len(self.feat_to_seq_id)}/{len(self.common_items)}")

    def get_feature(self, feat_name, attacked=False):
        """获取特征"""
        if attacked:
            return self.attacked_features.get(feat_name)
        else:
            return self.original_features.get(feat_name)

    def create_sample(self, user_id, target_feat_name, candidate_feat_names, use_attacked_for_target=False):
        """
        创建推荐样本

        使用特征文件名作为商品标识符，但在输入文本中使用映射后的序列ID
        """
        # 将特征文件名转换为序列ID用于模型输入
        target_seq_id = self.feat_to_seq_id.get(target_feat_name, target_feat_name)
        candidate_seq_ids = [self.feat_to_seq_id.get(f, f) for f in candidate_feat_names]

        # 构造输入文本
        candidates_text = ' {}, '.format('<extra_id_0> ' * self.image_feature_size_ratio).join(
            [str(c) for c in candidate_seq_ids]
        ) + ' <extra_id_0>' * self.image_feature_size_ratio

        source_text = f"We want to make recommendation for user_{user_id} .  Select the best item from these candidates : \n {candidates_text}"

        # 加载视觉特征
        feats = np.zeros((len(candidate_feat_names), self.image_feature_dim), dtype=np.float32)

        for i, feat_name in enumerate(candidate_feat_names):
            if feat_name == target_feat_name and use_attacked_for_target:
                feat = self.get_feature(feat_name, attacked=True)
            else:
                feat = self.get_feature(feat_name, attacked=False)

            if feat is not None:
                feats[i] = feat

        return {
            'source_text': source_text,
            'vis_feats': torch.from_numpy(feats),
            'target_feat_name': target_feat_name,
            'target_seq_id': target_seq_id,
            'candidate_feat_names': candidate_feat_names,
            'candidate_seq_ids': candidate_seq_ids,
        }


class VIP5Recommender:
    """VIP5推荐器"""

    def __init__(self, args, device='cuda'):
        self.args = args
        self.device = device if torch.cuda.is_available() else 'cpu'

        print(f"\nLoading VIP5 model on {self.device}...")

        config = create_config(args)

        self.tokenizer = P5Tokenizer.from_pretrained(
            args.backbone,
            max_length=args.max_text_length,
            do_lower_case=args.do_lower_case
        )

        self.model = VIP5Tuning.from_pretrained(
            args.backbone,
            config=config
        )
        self.model.to(self.device)
        self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        self.model.tokenizer = self.tokenizer

        if hasattr(args, 'checkpoint') and args.checkpoint and os.path.exists(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}")
            state_dict = load_state_dict(args.checkpoint, 'cpu')
            results = self.model.load_state_dict(state_dict, strict=False)
            print(f"Checkpoint loaded!")

        self.model.eval()
        print("VIP5 model ready!")

    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁') or tokenized_text[i] == '<extra_id_0>':
                curr += 1
            whole_word_ids.append(curr)
        return whole_word_ids[:len(input_ids) - 1] + [0]

    def prepare_batch(self, sample):
        source_text = sample['source_text']
        vis_feats = sample['vis_feats']

        input_ids = self.tokenizer.encode(
            source_text, padding=True, truncation=True,
            max_length=self.args.max_text_length
        )
        tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids = self.calculate_whole_word_ids(tokenized_text, input_ids)
        category_ids = [1 if token_id == 32099 else 0 for token_id in input_ids]

        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
        whole_word_ids = torch.LongTensor(whole_word_ids).unsqueeze(0).to(self.device)
        category_ids = torch.LongTensor(category_ids).unsqueeze(0).to(self.device)
        vis_feats = vis_feats.unsqueeze(0).to(self.device)

        return {
            'input_ids': input_ids,
            'whole_word_ids': whole_word_ids,
            'category_ids': category_ids,
            'vis_feats': vis_feats,
            'task': ['direct'],
        }

    @torch.no_grad()
    def get_recommendations(self, sample, num_beams=20, num_return=20):
        batch = self.prepare_batch(sample)

        beam_outputs = self.model.generate(
            input_ids=batch['input_ids'],
            whole_word_ids=batch['whole_word_ids'],
            category_ids=batch['category_ids'],
            vis_feats=batch['vis_feats'],
            task=batch['task'][0],
            max_length=50,
            num_beams=num_beams,
            no_repeat_ngram_size=0,
            num_return_sequences=num_return,
            early_stopping=True
        )

        generated_sents = self.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)

        recommended_items = []
        for sent in generated_sents:
            item_id = sent.strip()
            if item_id and item_id not in recommended_items:
                recommended_items.append(item_id)

        return recommended_items

    def get_item_rank(self, recommended_items, target_id):
        target_str = str(target_id)
        try:
            rank = recommended_items.index(target_str) + 1
            return rank
        except ValueError:
            return len(recommended_items) + 1


def evaluate_fixed(args, num_samples=100, num_candidates=20, verbose_samples=5):
    """修复版评估函数"""

    print("\n" + "="*70)
    print("VIP5 Attack Evaluation (Fixed Version)")
    print("="*70)

    # 加载数据集
    dataset = FixedEvaluationDataset(args, args.split)

    if len(dataset.available_items) < num_candidates:
        print(f"\nError: Not enough available items ({len(dataset.available_items)} < {num_candidates})")
        print("Please ensure attack and feature extraction were run correctly.")
        return None

    # 加载模型
    recommender = VIP5Recommender(args)

    # 准备评估
    random.seed(42)
    np.random.seed(42)

    # 获取有用户历史的商品
    items_in_sequences = set()
    for user, items in dataset.user_items.items():
        items_in_sequences.update(items)

    # 找到同时在特征集和用户序列中的商品
    evaluable_items = []
    for feat_name in dataset.available_items:
        seq_id = dataset.feat_to_seq_id.get(feat_name)
        if seq_id and seq_id in items_in_sequences:
            evaluable_items.append(feat_name)

    print(f"\nItems evaluable (in both features and sequences): {len(evaluable_items)}")

    if len(evaluable_items) < num_candidates:
        print("Warning: Not enough evaluable items, using all available items")
        evaluable_items = dataset.available_items

    # 选择用户
    users = list(dataset.user_items.keys())
    test_users = random.sample(users, min(num_samples * 2, len(users)))  # 多采样一些以应对跳过

    print(f"\n{'='*70}")
    print(f"Evaluating up to {num_samples} samples with {num_candidates} candidates each")
    print(f"{'='*70}")

    results = []

    for user_id in tqdm(test_users, desc="Evaluating"):
        if len(results) >= num_samples:
            break

        user_seq = dataset.user_items.get(user_id, [])
        if len(user_seq) < 2:
            continue

        # 选择目标商品 - 从用户序列末尾选择，且必须在可用特征集中
        target_seq_id = None
        target_feat_name = None

        for item in reversed(user_seq):
            if item in dataset.seq_id_to_feat:
                target_seq_id = item
                target_feat_name = dataset.seq_id_to_feat[item]
                break

        if target_feat_name is None:
            continue

        # 选择候选商品
        candidate_pool = [f for f in evaluable_items if f != target_feat_name]
        if len(candidate_pool) < num_candidates - 1:
            continue

        negative_samples = random.sample(candidate_pool, num_candidates - 1)
        candidate_feat_names = negative_samples + [target_feat_name]
        random.shuffle(candidate_feat_names)

        # 原始特征评估
        sample_original = dataset.create_sample(
            user_id=user_id,
            target_feat_name=target_feat_name,
            candidate_feat_names=candidate_feat_names,
            use_attacked_for_target=False
        )
        original_recs = recommender.get_recommendations(sample_original)
        original_rank = recommender.get_item_rank(original_recs, target_seq_id)

        # 攻击特征评估
        sample_attacked = dataset.create_sample(
            user_id=user_id,
            target_feat_name=target_feat_name,
            candidate_feat_names=candidate_feat_names,
            use_attacked_for_target=True
        )
        attacked_recs = recommender.get_recommendations(sample_attacked)
        attacked_rank = recommender.get_item_rank(attacked_recs, target_seq_id)

        rank_change = original_rank - attacked_rank

        result = {
            'user_id': user_id,
            'target_feat_name': target_feat_name,
            'target_seq_id': target_seq_id,
            'original_rank': original_rank,
            'attacked_rank': attacked_rank,
            'rank_change': rank_change,
        }
        results.append(result)

        if len(results) <= verbose_samples:
            print(f"\n  Sample {len(results)}:")
            print(f"  User: {user_id}, Target: {target_seq_id}")
            print(f"  Original Rank: {original_rank}, Attacked Rank: {attacked_rank}")
            print(f"  Rank Change: {rank_change:+d}")

    if len(results) == 0:
        print("\nNo valid samples for evaluation!")
        return None

    # 统计
    rank_changes = [r['rank_change'] for r in results]
    improved = sum(1 for x in rank_changes if x > 0)
    degraded = sum(1 for x in rank_changes if x < 0)
    unchanged = sum(1 for x in rank_changes if x == 0)

    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"\n[Settings]")
    print(f"  Total samples evaluated: {len(results)}")
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

    # 保存结果
    output_dir = SCRIPT_DIR / 'attack_evaluation_fixed'
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            'summary': {
                'total_samples': len(results),
                'mean_rank_change': float(np.mean(rank_changes)),
                'success_rate': improved / len(results),
                'improved': improved,
                'degraded': degraded,
                'unchanged': unchanged,
            },
            'detailed_results': results
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description='VIP5 Attack Evaluation - Fixed Version')

    parser.add_argument('--mode', type=str, default='diagnose',
                        choices=['diagnose', 'evaluate'],
                        help='Mode: diagnose or evaluate')
    parser.add_argument('--split', type=str, default='toys')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--num_candidates', type=int, default=20)
    parser.add_argument('--checkpoint', type=str, default=None)

    cmd_args = parser.parse_args()

    if cmd_args.mode == 'diagnose':
        diagnose_mapping(cmd_args.split)

    elif cmd_args.mode == 'evaluate':
        args = create_args(split=cmd_args.split, checkpoint_path=cmd_args.checkpoint)

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        if torch.cuda.is_available():
            cudnn.benchmark = True

        evaluate_fixed(
            args,
            num_samples=cmd_args.num_samples,
            num_candidates=cmd_args.num_candidates
        )


if __name__ == '__main__':
    main()
