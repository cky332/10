#!/usr/bin/env python3
"""
VIP5 Typography Attack Evaluation - Based on VIP5 Model Output
===============================================================
严格基于VIP5模型的实际推荐输出来评估攻击效果

核心逻辑：
1. 加载VIP5模型
2. 构造推荐任务（用户 + 候选商品列表）
3. 分别使用原始特征和攻击特征进行推荐
4. 比较目标商品在VIP5输出中的排名变化

Usage:
    python evaluate_attack_vip5.py --split toys --num_samples 100
"""

import os
import sys
import re
import argparse
import pickle
import json
import numpy as np
import random
import gzip
from pathlib import Path
from collections import defaultdict
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


# ============================================================
# 工具函数
# ============================================================

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

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


# ============================================================
# 配置类
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
    
    # 视觉特征配置
    args.image_feature_type = 'vitb32'
    args.image_feature_size_ratio = 2
    args.image_feature_dim = 512
    
    # Adapter配置
    args.use_adapter = True
    args.reduction_factor = 8
    args.use_single_adapter = True
    args.use_vis_layer_norm = True
    args.add_adapter_cross_attn = True
    args.use_lm_head_adapter = False  # 设为False避免modeling_vip5.py中的bug
    
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
    
    # 检查点路径
    if checkpoint_path:
        args.checkpoint = checkpoint_path
    else:
        # 默认检查点路径
        args.checkpoint = str(SCRIPT_DIR / 'snap' / 'toys-vitb32-2-8-20' / 'BEST_EVAL_LOSS.pth')
    
    return args


def create_config(args):
    """创建VIP5模型配置 - 完全按照notebook的方式"""
    from adapters import AdapterConfig
    
    config = T5Config.from_pretrained(args.backbone)
    
    # 复制所有args到config
    for k, v in vars(args).items():
        setattr(config, k, v)
    
    # 图像特征配置
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
    
    # Dropout配置
    config.dropout_rate = args.dropout
    config.dropout = args.dropout
    config.attention_dropout = args.dropout
    config.activation_dropout = args.dropout
    
    config.losses = args.losses
    
    # 从losses解析tasks列表 - 这是关键！
    tasks = re.split("[, ]+", args.losses)
    
    if args.use_adapter:
        config.adapter_config = AdapterConfig()
        config.adapter_config.tasks = tasks  # 关键：设置tasks属性
        config.adapter_config.d_model = config.d_model
        config.adapter_config.use_single_adapter = args.use_single_adapter
        config.adapter_config.reduction_factor = args.reduction_factor
        config.adapter_config.track_z = False
    else:
        config.adapter_config = None
    
    return config


# ============================================================
# 数据加载类
# ============================================================

class AttackEvaluationDataset:
    """攻击评估数据集"""
    
    def __init__(self, args, split='toys'):
        self.args = args
        self.split = split
        self.image_feature_type = args.image_feature_type
        self.image_feature_dim = args.image_feature_dim
        self.image_feature_size_ratio = args.image_feature_size_ratio
        
        # 加载数据
        data_dir = SCRIPT_DIR / 'data' / split
        
        # 用户交互序列
        self.sequential_data = ReadLineFromFile(str(data_dir / 'sequential_data.txt'))
        
        # 用户-物品映射
        item_count = defaultdict(int)
        self.user_items = defaultdict()
        for line in self.sequential_data:
            user, items = line.strip().split(' ', 1)
            items = [int(item) for item in items.split(' ')]
            self.user_items[user] = items
            for item in items:
                item_count[item] += 1
        
        self.all_item = list(item_count.keys())
        
        # 数据映射
        datamaps = load_json(str(data_dir / 'datamaps.json'))
        self.user2id = datamaps['user2id']
        self.item2id = datamaps['item2id']
        self.id2item = datamaps['id2item']  # 数字ID -> ASIN
        self.user_list = list(datamaps['user2id'].keys())
        self.item_list = list(datamaps['item2id'].keys())
        
        # 用户名称
        self.user_id2name = load_pickle(str(data_dir / 'user_id2name.pkl'))
        
        # 关键：加载 ASIN -> 图片文件名 的映射
        item2img_path = data_dir / 'item2img_dict.pkl'
        if item2img_path.exists():
            self.item2img_dict = load_pickle(str(item2img_path))
            print(f"Loaded item2img_dict with {len(self.item2img_dict)} items")
            # 打印示例
            sample_items = list(self.item2img_dict.items())[:3]
            print(f"Sample item2img mapping: {sample_items}")
        else:
            print(f"Warning: item2img_dict.pkl not found at {item2img_path}")
            print("Will try to build mapping from feature files...")
            self.item2img_dict = {}

        print(f"Loaded {len(self.user_list)} users, {len(self.item_list)} items")

        # 关键修复：建立反向映射 图片文件名 -> 数字ID
        self._build_reverse_mapping()

    def _build_reverse_mapping(self):
        """
        建立反向映射：图片文件名 -> 数字ID

        这是关键修复！原来的映射方向是：
            数字ID -> ASIN -> item2img_dict -> 图片文件名
        但 id2item 的 value (ASIN) 和 item2img_dict 的 key 格式可能不同。

        正确的做法是建立反向映射：
            图片文件名 -> item2img_dict的key -> item2id -> 数字ID
        """
        self.imgname_to_itemid = {}  # 图片文件名 -> 数字ID
        self.itemid_to_imgname = {}  # 数字ID -> 图片文件名

        # 建立 图片文件名 -> item2img_dict的key 的反向映射
        imgname_to_asin = {}
        for asin_key, img_info in self.item2img_dict.items():
            # 解析图片文件名
            if isinstance(img_info, str):
                img_filename = img_info
            elif isinstance(img_info, list) and len(img_info) > 0:
                img_filename = img_info[0]
            elif isinstance(img_info, dict) and 'image' in img_info:
                img_filename = img_info['image']
            else:
                continue

            # 去掉路径和扩展名
            if '/' in img_filename:
                img_filename = img_filename.split('/')[-1]
            if '.' in img_filename:
                img_filename = img_filename.rsplit('.', 1)[0]

            imgname_to_asin[img_filename] = asin_key

        print(f"\nBuilding reverse mapping...")
        print(f"  imgname_to_asin entries: {len(imgname_to_asin)}")

        # 建立 图片文件名 -> 数字ID 的映射
        # item2img_dict 的 key 应该与 item2id 的 key 相同
        matched = 0
        for img_filename, asin_key in imgname_to_asin.items():
            # asin_key 是 item2img_dict 的 key，应该也是 item2id 的 key
            if asin_key in self.item2id:
                numeric_id = self.item2id[asin_key]
                self.imgname_to_itemid[img_filename] = int(numeric_id)
                self.itemid_to_imgname[int(numeric_id)] = img_filename
                matched += 1

        print(f"  Successfully mapped: {matched} items")
        print(f"  imgname_to_itemid entries: {len(self.imgname_to_itemid)}")

        # 打印示例
        if len(self.imgname_to_itemid) > 0:
            sample = list(self.imgname_to_itemid.items())[:3]
            print(f"  Sample mapping: {sample}")

    def get_image_filename(self, item_id):
        """
        根据数字ID获取图片文件名
        数字ID -> ASIN -> 图片文件名
        """
        # 先转换为ASIN
        asin = self.id2item.get(str(item_id), None)
        if asin is None:
            return None
        
        # 再转换为图片文件名
        if asin in self.item2img_dict:
            img_info = self.item2img_dict[asin]
            
            # 处理不同的格式
            if isinstance(img_info, str):
                img_filename = img_info
            elif isinstance(img_info, list) and len(img_info) > 0:
                img_filename = img_info[0]  # 如果是列表，取第一个
            elif isinstance(img_info, dict) and 'image' in img_info:
                img_filename = img_info['image']
            else:
                return None
            
            # 去掉扩展名和路径
            if '/' in img_filename:
                img_filename = img_filename.split('/')[-1]
            if img_filename.endswith('.jpg') or img_filename.endswith('.png') or img_filename.endswith('.jpeg'):
                img_filename = img_filename.rsplit('.', 1)[0]
            
            return img_filename
        
        return None
    
    def get_feature_path(self, item_id, attacked=False):
        """获取特征文件路径"""
        if attacked:
            feat_dir = SCRIPT_DIR / 'features' / f'{self.image_feature_type}_features' / f'{self.split}_attacked'
        else:
            feat_dir = SCRIPT_DIR / 'features' / f'{self.image_feature_type}_features' / f'{self.split}_original'

        # 优先使用反向映射获取图片文件名
        item_id_int = int(item_id)
        img_filename = self.itemid_to_imgname.get(item_id_int)

        # 如果反向映射没有，尝试正向映射
        if img_filename is None:
            img_filename = self.get_image_filename(item_id)

        if img_filename is None:
            return None

        return feat_dir / f'{img_filename}.npy'
    
    def load_feature(self, item_id, attacked=False):
        """加载单个物品的特征"""
        feat_path = self.get_feature_path(item_id, attacked)
        if feat_path is not None and feat_path.exists():
            return np.load(feat_path)
        else:
            return None
    
    def get_available_items(self, common_feature_files):
        """
        获取可用的商品列表（同时在数据集中且有特征文件）

        修复版：使用反向映射，从特征文件名出发找到数字ID

        Args:
            common_feature_files: 同时有原始和攻击特征的文件名集合

        Returns:
            list: 可用的商品数字ID列表
        """
        available_items = []
        matched_via_reverse = 0
        matched_via_forward = 0
        no_mapping = 0

        # 方法1：使用反向映射（推荐）
        # 从特征文件名直接找到数字ID
        for feat_name in common_feature_files:
            if feat_name in self.imgname_to_itemid:
                item_id = self.imgname_to_itemid[feat_name]
                if item_id not in available_items:
                    available_items.append(item_id)
                    matched_via_reverse += 1

        # 方法2：使用原来的正向映射作为补充
        # 数字ID -> ASIN -> 图片文件名
        for item_id in self.id2item.keys():
            item_id_int = int(item_id)
            if item_id_int in available_items:
                continue  # 已经通过反向映射找到了

            img_filename = self.get_image_filename(item_id_int)
            if img_filename and img_filename in common_feature_files:
                available_items.append(item_id_int)
                matched_via_forward += 1

        print(f"\n--- Debug: get_available_items ---")
        print(f"Total feature files to match: {len(common_feature_files)}")
        print(f"Matched via reverse mapping (imgname->itemid): {matched_via_reverse}")
        print(f"Matched via forward mapping (itemid->imgname): {matched_via_forward}")
        print(f"Total available items: {len(available_items)}")

        # 显示示例
        if len(available_items) > 0:
            sample_id = available_items[0]
            sample_img = self.itemid_to_imgname.get(sample_id, "N/A")
            print(f"\nSample: item_id={sample_id}, img_filename={sample_img}")

        # 如果仍然没有匹配，显示调试信息
        if len(available_items) == 0:
            print(f"\nNo items matched! Debug info:")
            sample_features = list(common_feature_files)[:5]
            print(f"  Sample feature file names: {sample_features}")
            sample_imgname = list(self.imgname_to_itemid.keys())[:5] if self.imgname_to_itemid else []
            print(f"  Sample imgname_to_itemid keys: {sample_imgname}")

        return available_items
    
    def create_recommendation_sample(self, user_id, target_item_id, candidate_ids, 
                                      use_attacked_for_target=False):
        """
        创建一个推荐样本
        
        Args:
            user_id: 用户ID
            target_item_id: 目标商品ID（我们想让它被推荐）
            candidate_ids: 候选商品ID列表
            use_attacked_for_target: 是否对目标商品使用攻击后的特征
        
        Returns:
            batch: 可直接输入VIP5模型的batch
        """
        # 构造输入文本（B-8模板）
        candidates_text = ' {}, '.format('<extra_id_0> ' * self.image_feature_size_ratio).join(
            [str(c) for c in candidate_ids]
        ) + ' <extra_id_0>' * self.image_feature_size_ratio
        
        source_text = f"We want to make recommendation for user_{user_id} .  Select the best item from these candidates : \n {candidates_text}"
        
        # 加载视觉特征
        feats = np.zeros((len(candidate_ids), self.image_feature_dim), dtype=np.float32)
        
        for i, cand_id in enumerate(candidate_ids):
            if str(cand_id) == str(target_item_id) and use_attacked_for_target:
                # 对目标商品使用攻击特征
                feat = self.load_feature(cand_id, attacked=True)
                if feat is None:
                    # 如果攻击特征不存在，使用原始特征
                    feat = self.load_feature(cand_id, attacked=False)
            else:
                # 其他商品使用原始特征
                feat = self.load_feature(cand_id, attacked=False)
            
            if feat is not None:
                feats[i] = feat
        
        return {
            'source_text': source_text,
            'vis_feats': torch.from_numpy(feats),
            'target_item_id': target_item_id,
            'candidate_ids': candidate_ids,
        }


# ============================================================
# VIP5推荐器
# ============================================================

class VIP5Recommender:
    """VIP5推荐器封装"""
    
    def __init__(self, args, device='cuda'):
        self.args = args
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"\nLoading VIP5 model on {self.device}...")
        
        # 创建配置
        config = create_config(args)
        
        # 加载tokenizer
        self.tokenizer = P5Tokenizer.from_pretrained(
            args.backbone,
            max_length=args.max_text_length,
            do_lower_case=args.do_lower_case
        )
        
        # 使用from_pretrained创建模型（与notebook一致）
        print(f"Building Model from {args.backbone}")
        self.model = VIP5Tuning.from_pretrained(
            args.backbone,
            config=config
        )
        self.model.to(self.device)
        
        # 调整token embeddings
        self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        self.model.tokenizer = self.tokenizer
        
        # 加载检查点
        if hasattr(args, 'checkpoint') and args.checkpoint and os.path.exists(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}")
            state_dict = load_state_dict(args.checkpoint, 'cpu')
            results = self.model.load_state_dict(state_dict, strict=False)
            print(f"Checkpoint loaded! {results}")
        else:
            print("Warning: No checkpoint found, using pretrained weights only!")
        
        self.model.eval()
        print("VIP5 model ready!")
    
    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        """计算whole word IDs"""
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁') or tokenized_text[i] == '<extra_id_0>':
                curr += 1
            whole_word_ids.append(curr)
        return whole_word_ids[:len(input_ids) - 1] + [0]
    
    def prepare_batch(self, sample):
        """准备模型输入batch"""
        source_text = sample['source_text']
        vis_feats = sample['vis_feats']
        
        # Tokenize
        input_ids = self.tokenizer.encode(
            source_text, padding=True, truncation=True, 
            max_length=self.args.max_text_length
        )
        tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids = self.calculate_whole_word_ids(tokenized_text, input_ids)
        category_ids = [1 if token_id == 32099 else 0 for token_id in input_ids]
        
        # 转换为tensor
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
        """
        获取VIP5的推荐结果
        
        Returns:
            list: 按推荐顺序排列的商品ID列表
        """
        batch = self.prepare_batch(sample)
        
        # 使用beam search生成推荐
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
        
        # 解码生成结果
        generated_sents = self.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
        
        # 解析商品ID
        recommended_items = []
        for sent in generated_sents:
            try:
                item_id = int(sent.strip())
                if item_id not in recommended_items:
                    recommended_items.append(item_id)
            except:
                pass
        
        return recommended_items
    
    def get_item_rank(self, recommended_items, target_item_id):
        """获取目标商品在推荐列表中的排名"""
        target_id = int(target_item_id)
        try:
            rank = recommended_items.index(target_id) + 1
            return rank
        except ValueError:
            # 目标商品不在推荐列表中
            return len(recommended_items) + 1


# ============================================================
# 评估主函数
# ============================================================

def evaluate_attack(args, num_samples=100, num_candidates=20, verbose_samples=5):
    """
    评估Typography攻击效果
    
    Args:
        args: 模型参数
        num_samples: 评估样本数量
        num_candidates: 每个样本的候选商品数量
        verbose_samples: 详细打印的样本数量
    """
    
    print("\n" + "="*70)
    print("VIP5 Typography Attack Evaluation (Based on Model Output)")
    print("="*70)
    
    # 检查特征目录 (与 evaluate_attack.py 保持一致)
    original_feat_dir = SCRIPT_DIR / 'features' / f'{args.image_feature_type}_features' / f'{args.split}_original'
    attacked_feat_dir = SCRIPT_DIR / 'features' / f'{args.image_feature_type}_features' / f'{args.split}_attacked'
    
    if not original_feat_dir.exists():
        print(f"\nError: Original features not found at {original_feat_dir}")
        print("Please run feature extraction first!")
        return None
    
    if not attacked_feat_dir.exists():
        print(f"\nError: Attacked features not found at {attacked_feat_dir}")
        print("Please run attack and feature extraction first!")
        return None
    
    # 找出同时有原始和攻击特征的商品
    original_items = set(f.stem for f in original_feat_dir.glob('*.npy'))
    attacked_items = set(f.stem for f in attacked_feat_dir.glob('*.npy'))
    common_items = original_items & attacked_items
    
    print(f"\nOriginal features: {len(original_items)} items")
    print(f"Attacked features: {len(attacked_items)} items")
    print(f"Common items: {len(common_items)} items")
    
    if len(common_items) < 10:
        print("Error: Not enough common items for evaluation!")
        return None
    
    # 加载数据集
    dataset = AttackEvaluationDataset(args, args.split)
    
    # 加载VIP5模型
    recommender = VIP5Recommender(args)
    
    # 准备评估
    random.seed(42)
    np.random.seed(42)
    
    # 使用dataset的方法筛选可用商品
    available_items = dataset.get_available_items(common_items)
    
    print(f"Available items for evaluation: {len(available_items)}")
    
    if len(available_items) < num_candidates:
        print("Error: Not enough available items!")
        return None
    
    # 选择用户
    users = list(dataset.user_items.keys())
    test_users = random.sample(users, min(num_samples, len(users)))
    
    print(f"\n{'='*70}")
    print(f"Evaluating {len(test_users)} samples with {num_candidates} candidates each")
    print(f"{'='*70}")
    
    # 详细样本展示
    print(f"\n--- Detailed examples (first {verbose_samples}) ---")
    
    results = []
    
    for i, user_id in enumerate(tqdm(test_users, desc="Evaluating")):
        user_seq = dataset.user_items[user_id]
        
        # 选择目标商品（从用户历史中选择最后一个作为目标）
        if len(user_seq) < 2:
            continue
        
        target_item_id = user_seq[-1]

        # 检查目标商品是否有攻击特征
        # 优先使用反向映射，再尝试正向映射
        target_img_filename = dataset.itemid_to_imgname.get(target_item_id)
        if target_img_filename is None:
            target_img_filename = dataset.get_image_filename(target_item_id)
        if target_img_filename is None or target_img_filename not in common_items:
            continue
        
        # 选择候选商品（包括目标商品和负样本）
        negative_pool = [item for item in available_items 
                        if item not in user_seq and item != target_item_id]
        
        if len(negative_pool) < num_candidates - 1:
            continue
        
        negative_samples = random.sample(negative_pool, num_candidates - 1)
        candidate_ids = negative_samples + [target_item_id]
        random.shuffle(candidate_ids)
        
        # ===== 使用原始特征进行推荐 =====
        sample_original = dataset.create_recommendation_sample(
            user_id=user_id,
            target_item_id=target_item_id,
            candidate_ids=candidate_ids,
            use_attacked_for_target=False
        )
        
        original_recs = recommender.get_recommendations(sample_original)
        original_rank = recommender.get_item_rank(original_recs, target_item_id)
        
        # ===== 使用攻击特征进行推荐 =====
        sample_attacked = dataset.create_recommendation_sample(
            user_id=user_id,
            target_item_id=target_item_id,
            candidate_ids=candidate_ids,
            use_attacked_for_target=True
        )
        
        attacked_recs = recommender.get_recommendations(sample_attacked)
        attacked_rank = recommender.get_item_rank(attacked_recs, target_item_id)
        
        # 计算排名变化
        rank_change = original_rank - attacked_rank  # 正数 = 攻击后排名更好
        
        result = {
            'user_id': user_id,
            'target_item_id': target_item_id,
            'num_candidates': len(candidate_ids),
            'original_rank': original_rank,
            'attacked_rank': attacked_rank,
            'rank_change': rank_change,
            'original_in_top5': original_rank <= 5,
            'attacked_in_top5': attacked_rank <= 5,
        }
        results.append(result)
        
        # 详细打印
        if i < verbose_samples:
            print(f"\n  Sample {i+1}:")
            print(f"  User: {user_id}, Target Item: {target_item_id}")
            print(f"  Original Rank: {original_rank}, Attacked Rank: {attacked_rank}")
            print(f"  Rank Change: {rank_change:+d} ({'improved' if rank_change > 0 else 'degraded' if rank_change < 0 else 'unchanged'})")
            if len(original_recs) > 0:
                print(f"  Top-3 Original: {original_recs[:3]}")
            if len(attacked_recs) > 0:
                print(f"  Top-3 Attacked: {attacked_recs[:3]}")
    
    if len(results) == 0:
        print("\nNo valid samples for evaluation!")
        return None
    
    # ===== 统计汇总 =====
    rank_changes = [r['rank_change'] for r in results]
    
    improved = sum(1 for x in rank_changes if x > 0)
    degraded = sum(1 for x in rank_changes if x < 0)
    unchanged = sum(1 for x in rank_changes if x == 0)
    
    # Top-5效果
    entered_top5 = sum(1 for r in results if r['attacked_in_top5'] and not r['original_in_top5'])
    left_top5 = sum(1 for r in results if r['original_in_top5'] and not r['attacked_in_top5'])
    
    summary = {
        'total_samples': len(results),
        'num_candidates': num_candidates,
        'rank_change': {
            'mean': float(np.mean(rank_changes)),
            'std': float(np.std(rank_changes)),
            'median': float(np.median(rank_changes)),
            'max_improvement': int(np.max(rank_changes)),
            'max_degradation': int(np.min(rank_changes)),
        },
        'attack_effect': {
            'improved': improved,
            'degraded': degraded,
            'unchanged': unchanged,
            'success_rate': improved / len(results),
        },
        'top5_effect': {
            'entered_top5': entered_top5,
            'left_top5': left_top5,
        }
    }
    
    # ===== 打印结果 =====
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS (Based on VIP5 Model Output)")
    print(f"{'='*70}")
    
    print(f"\n[Settings]")
    print(f"  Total samples evaluated: {summary['total_samples']}")
    print(f"  Candidates per sample: {num_candidates}")
    
    print(f"\n[Rank Changes] (positive = attack helped, target item ranked higher)")
    print(f"  Mean:   {summary['rank_change']['mean']:+.2f}")
    print(f"  Median: {summary['rank_change']['median']:+.2f}")
    print(f"  Std:    {summary['rank_change']['std']:.2f}")
    print(f"  Best:   {summary['rank_change']['max_improvement']:+d} positions")
    print(f"  Worst:  {summary['rank_change']['max_degradation']:+d} positions")
    
    print(f"\n[Attack Success Rate]")
    print(f"  Improved (attack helped): {improved} ({improved/len(results)*100:.1f}%)")
    print(f"  Degraded (attack hurt):   {degraded} ({degraded/len(results)*100:.1f}%)")
    print(f"  Unchanged:                {unchanged} ({unchanged/len(results)*100:.1f}%)")
    
    print(f"\n[Top-5 Effect]")
    print(f"  Newly entered Top-5 after attack: {entered_top5}")
    print(f"  Left Top-5 after attack: {left_top5}")
    
    # ===== 分析 =====
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    
    if summary['attack_effect']['success_rate'] > 0.5:
        print("\n✓ Attack is EFFECTIVE on VIP5 model!")
        print(f"  Success rate: {summary['attack_effect']['success_rate']*100:.1f}%")
        print(f"  Average rank improvement: {summary['rank_change']['mean']:+.2f}")
    elif summary['attack_effect']['success_rate'] > 0.3:
        print("\n⚠ Attack has MODERATE effect on VIP5 model")
        print(f"  Success rate: {summary['attack_effect']['success_rate']*100:.1f}%")
    else:
        print("\n✗ Attack is NOT effective on VIP5 model")
        print(f"  Success rate: {summary['attack_effect']['success_rate']*100:.1f}%")
        print("\nPossible reasons:")
        print("  1. Typography text disrupts important visual features")
        print("  2. VIP5's multi-modal fusion reduces visual attack impact")
        print("  3. Text-based features dominate the recommendation")
        print("\nSuggestions:")
        print("  1. Use gradient-based adversarial perturbation")
        print("  2. Attack text modality instead of visual modality")
        print("  3. Use more subtle visual modifications")
    
    # 保存结果
    output_dir = SCRIPT_DIR / 'attack_evaluation_vip5'
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump({
            'summary': summary,
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    return summary, results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Typography Attack on VIP5 (Based on Model Output)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluate_attack_vip5.py --split toys --num_samples 100
  
  # More samples with larger candidate pool
  python evaluate_attack_vip5.py --split toys --num_samples 200 --num_candidates 50
  
  # Specify checkpoint path
  python evaluate_attack_vip5.py --split toys --checkpoint path/to/BEST.pth
        """
    )
    
    parser.add_argument('--split', type=str, default='toys',
                        help='Dataset split (default: toys)')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of evaluation samples (default: 100)')
    parser.add_argument('--num_candidates', type=int, default=20,
                        help='Number of candidates per sample (default: 20)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--verbose_samples', type=int, default=5,
                        help='Number of samples to print in detail')
    
    cmd_args = parser.parse_args()
    
    # 创建模型参数
    args = create_args(split=cmd_args.split, checkpoint_path=cmd_args.checkpoint)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if torch.cuda.is_available():
        cudnn.benchmark = True
    
    # 运行评估
    evaluate_attack(
        args, 
        num_samples=cmd_args.num_samples,
        num_candidates=cmd_args.num_candidates,
        verbose_samples=cmd_args.verbose_samples
    )


if __name__ == '__main__':
    main()