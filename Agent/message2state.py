import numpy as np

CARD_ORDER = ['H2', 'C2', 'S2', 'D2', 'H3', 'C3', 'S3', 'D3', 'H4', 'C4', 'S4', 'D4',
              'H5', 'C5', 'S5', 'D5', 'H6', 'C6', 'S6', 'D6', 'H7', 'C7', 'S7', 'D7',
              'H8', 'C8', 'S8', 'D8', 'H9', 'C9', 'S9', 'D9', 'HT', 'CT', 'ST', 'DT',
              'HJ', 'CJ', 'SJ', 'DJ', 'HQ', 'CQ', 'SQ', 'DQ', 'HK', 'CK', 'SK', 'DK',
              'HA', 'CA', 'SA', 'DA', 'HB', 'SR']


def cards_to_vector(cards, card_order):
    """
    把手牌列表映射为固定顺序的计数向量
    :param cards: list[str]  输入的牌，比如 ['HA','DA','DA']
    :param card_order: list[str]  固定顺序，比如 ['H2','C2',...,'HA','DA']
    :return: numpy.ndarray, shape=(len(card_order),), dtype=int
    """
    vec = np.zeros(len(card_order), dtype=int)
    index_map = {c: i for i, c in enumerate(card_order)}  # 建立映射
    for card in cards:
        if card in index_map:
            vec[index_map[card]] += 1
    return vec


def convert_message_to_state(origin_cards, played_cards, up_player_played, teammate_played, others_played1
                             , others_played2, others_played3, remaining_counts_others, wild_cards):
    origin_cards_vec = cards_to_vector(origin_cards, CARD_ORDER)
    played_cards_vec = cards_to_vector(played_cards, CARD_ORDER)
    up_player_played_vec = cards_to_vector(up_player_played, CARD_ORDER)
    teammate_played_vec = cards_to_vector(teammate_played, CARD_ORDER)
    others_played1_vec = cards_to_vector(others_played1, CARD_ORDER)
    others_played2_vec = cards_to_vector(others_played2, CARD_ORDER)
    others_played3_vec = cards_to_vector(others_played3, CARD_ORDER)

    # 生成初始牌池向量：前 (N-2) 位为4，最后 2 位为2
    N = len(CARD_ORDER)
    init_vec = np.full(N, 4.0, dtype=np.float32)
    init_vec[-2:] = 2.0

    # 一行减去已知的牌（origin + played + others1 + others2 + others3）
    unknown_cards_vec = init_vec - (
        origin_cards_vec + played_cards_vec +
        others_played1_vec + others_played2_vec + others_played3_vec
    )

    # 防止负值（如果某个计数被多次减到负数，裁剪到0）
    unknown_cards_vec = np.clip(unknown_cards_vec, 0.0, None)

    remaining_counts_others = np.atleast_1d(np.array(remaining_counts_others, dtype=np.float32))
    wild_cards = np.atleast_1d(np.array(wild_cards, dtype=np.float32))
    # 把所有向量合并
    state = np.concatenate([
        origin_cards_vec,
        played_cards_vec,
        up_player_played_vec,
        teammate_played_vec,
        others_played1_vec,
        others_played2_vec,
        others_played3_vec,
        remaining_counts_others,
        unknown_cards_vec,
        wild_cards
    ])


    # # --- 动作 mask ---
    # action_mask = np.zeros(1000, dtype=np.float32)
    # for a in actions:
    #     try:
    #         idx = int(a["index"])  # 这里强制转 int
    #     except Exception:
    #         continue
    #     if 0 <= idx < 1000:
    #         action_mask[idx] = 1

    return state

from typing import List, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

############################################
# 牌型与点数定义
############################################
# 点数从小到大：3..2，再加两王 B(小王), R(大王)
RANKS = ['3','4','5','6','7','8','9','T','J','Q','K','A','2','B','R']
RANK2ID = {r:i for i, r in enumerate(RANKS)}   # -> 0..14

# 牌型
ACTION_TYPES = ['PASS','Single','Pair','Trips','ThreePair','ThreeWithTwo','TwoTrips','Straight','StraightFlush','Bomb']
TYPE2ID = {t:i for i, t in enumerate(ACTION_TYPES)}  # -> 0..9

TYPE_DIM = len(ACTION_TYPES)     # 10
RANK_DIM = len(RANKS)            # 15
FEAT_DIM = TYPE_DIM + RANK_DIM   # “牌型+点数”

############################################
# 从 action['action'] 提取 (type, rank) 两个“符号”
############################################
def extract_type_and_rank(a: Dict) -> (str, Optional[str]):
    """
    期望输入格式示例：
      {'action': ['Single','2',['S2']], 'index': 7}
      {'action': ['Pair','7',['C7','H7']], 'index': 11}
      {'action': ['PASS','PASS','PASS'], 'index': 0}

    返回 (atype, rank_token)
      - atype: 'PASS'/'Single'/'Pair'/...
      - rank_token: '3'..'2'/'SB'/'HR' 或 None
    规则：
      1) 优先使用 action['action'][1] 作为“点数标签”（若它属于 {RANKS}）
      2) 否则尝试从牌列表里解析（如 ['B'] / ['R'] / ['S2'] -> 'B'/'R'/'2'）
    """
    raw = a['action']
    atype = str(raw[0])

    # PASS 没点数
    if atype == 'PASS':
        return atype, None
    else:
        cand = str(raw[1])
        return atype, cand

############################################
# 单条 action -> “牌型+点数” one-hot 特征
############################################
def action_to_simple_feat(a: Dict) -> np.ndarray:
    """
    输出长度 FEAT_DIM = TYPE_DIM + RANK_DIM 的 one-hot 拼接
      [ type_onehot(9) , rank_onehot(15) ]
    """
    atype, rank = extract_type_and_rank(a)

    type_oh = np.zeros(TYPE_DIM, dtype=np.float32)
    type_oh[TYPE2ID.get(atype)] = 1.0

    rank_oh = np.zeros(RANK_DIM, dtype=np.float32)
    if rank is not None:
        rank_oh[RANK2ID[rank]] = 1.0

    return np.concatenate([type_oh, rank_oh], axis=0)  # (FEAT_DIM,)

############################################
# 构造“语义掩码” bundle：加法掩码 + (type+rank)特征
############################################
def build_semantic_bundle(
    actions: List[Dict],
    action_dim: int,
    illegal_value: float = -1e9,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    返回：
      {
        'mask': (A,)   float，加法掩码（合法=0, 非法=illegal_value）
        'feat': (A,F)  float，A=action_dim, F=FEAT_DIM，仅含“牌型+点数”特征
      }
    说明：
      - 对非法 index 的 feat 填 0（反正被掩掉）
      - 越界 index 自动忽略
    """
    A = action_dim
    add_mask = torch.full((A,), illegal_value, dtype=torch.float32, device=device)
    feat = torch.zeros((A, FEAT_DIM), dtype=torch.float32, device=device)

    for item in actions:
        idx = int(item['index'])
        if 0 <= idx < A:
            add_mask[idx] = 0.0
            f = action_to_simple_feat(item)          # (F,)
            feat[idx] = torch.from_numpy(f)

    return {'mask': add_mask, 'feat': feat}

#===============================================transition传输=================================================================
def build_sparse_semantic(actions: list, action_dim: int):
    """
    生成“稀疏语义包”（用于网络传输/存储 transition）。
    返回字典：
      {
        'action_dim': int,
        'indices':  [i1, i2, ...],               # 合法 index（升序、去重、过滤越界）
        'type_ids': [t1, t2, ...],               # 与 indices 一一对应
        'rank_ids': [r1, r2, ...],               # 与 indices 一一对应，None 用 -1
        'ver': 1, 'rule': 'type+rank_v1'         # 可选：版本/规则
      }
    """
    allowed = []
    type_ids, rank_ids = [], []

    seen = set()
    for item in actions:
        idx = int(item['index'])
        if 0 <= idx < action_dim and idx not in seen:
            seen.add(idx)
            atype, rank = extract_type_and_rank(item)
            allowed.append(idx)
            type_ids.append(TYPE2ID.get(atype))
            rank_ids.append(RANK2ID[rank] if rank in RANK2ID else -1)

    allowed_sorted = sorted(range(len(allowed)), key=lambda k: allowed[k])
    indices = [allowed[i] for i in allowed_sorted]
    type_ids = [type_ids[i] for i in allowed_sorted]
    rank_ids = [rank_ids[i] for i in allowed_sorted]

    return {
        'action_dim': int(action_dim),
        'indices': indices,
        'type_ids': type_ids,
        'rank_ids': rank_ids,
    }

def sparse_to_dense(sparse_list, action_dim=500, illegal_value: float = -1e9):
    """
    批量版：将长度 B 的 sparse_list 转成稠密 numpy：
      masks_np: (B, A) float32，加法掩码（合法=0，非法=illegal_value）
      feats_np: (B, A, FEAT_DIM) float32，非法行全 0

    支持两种条目格式（可混用，但建议统一）：
      1) 标准稀疏结构 dict：
         {'indices': [...], 'type_ids': [...], 'rank_ids': [...], 'action_dim': A, ...}
         （可选）也支持已有稠密：{'mask': (A,), 'feat': (A,FEAT_DIM)} ——会直接使用

      2) “原始服务端 actions 列表”：
         [ {'action': [...], 'index': int}, ... ]
         这种会在函数内部自动转成稀疏结构（需要 action_dim 参数）

    参数：
      sparse_list: list，长度 B
      action_dim: 若条目没有 'action_dim' 字段，或者是原始 actions 列表，则必须提供
      illegal_value: 非法位掩码值（默认 -1e9）

    返回：
      (masks_np, feats_np)
    """
    B = len(sparse_list)
    if B == 0:
        raise ValueError("sparse_to_dense: empty sparse_list")

    # ---- 推断/检查 A（action_dim） ----
    A = 500

    # 初始化输出
    masks_np = np.full((B, A), illegal_value, dtype=np.float32)
    feats_np = np.zeros((B, A, FEAT_DIM), dtype=np.float32)

    for b, item in enumerate(sparse_list):

        # --- 情况 B：标准稀疏结构 ---
        if isinstance(item, dict) and ('indices' in item and 'type_ids' in item and 'rank_ids' in item):
            indices  = item['indices']
            type_ids = item['type_ids']
            rank_ids = item['rank_ids']
            # 如果条目里也有 action_dim，校验一下
            if 'action_dim' in item and int(item['action_dim']) != A:
                raise ValueError(f"[sparse_to_dense] row {b}: action_dim mismatch: {item['action_dim']} vs {A}")

        # 填充该样本的 mask/feat
        for idx, tid, rid in zip(indices, type_ids, rank_ids):
            if 0 <= idx < A:
                masks_np[b, idx] = 0.0
                if 0 <= tid < TYPE_DIM:
                    feats_np[b, idx, tid] = 1.0
                if 0 <= rid < RANK_DIM:
                    feats_np[b, idx, TYPE_DIM + rid] = 1.0
            # 越界 index 自动忽略（也可以在这里打印 warn）

    return masks_np, feats_np