"""
message_to_observation.py

提供:
- card_index_map: 全局 54 张牌到 index 的映射（H2 C2 S2 D2 H3 ... A + 小王 大王）
- convert_cards_to_54(cards_list) -> np.array shape (54,)  (计数: {0,1,2})
- message_to_observation(message: dict, K_history=8, MAX_ACTIONS=64, target_dim=513) ->
      {
        'obs': {structured dict...},          # 可读结构化 observation
        'obs_vector': np.array(shape=(target_dim,))  # 固定长度向量（zero-pad/truncate）
      }

假设 message 来源于你的 WebSocket 客户端，且 message 已经是 JSON -> dict 的结果（即接收代码已做 json.loads）。
Supported operations: "Deal", "RequestAction", "PlayCard", "GameResult" （按你给的说明书）
"""
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

# -------------------------
# 全局定义：牌面顺序（54维）
# -------------------------
SUITS = ['H', 'C', 'S', 'D']  # 红桃/梅花/黑桃/方片
RANKS = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']  # 注意顺序 2..A
# 54张：按 rank 外圈，suit 内圈排列: H2 C2 S2 D2 H3 C3 S3 D3 ... HA CA SA DA, 然后小王 HB, 大王 SR
CARD_ORDER = []
for r in RANKS:
    for s in SUITS:
        CARD_ORDER.append(s + r)  # e.g. 'H2','C2','S2','D2','H3',...
# append small joker (B) then big joker (R) - 这里按你描述 "HB SR"
CARD_ORDER.append('HB')  # 小王
CARD_ORDER.append('SR')  # 大王
assert len(CARD_ORDER) == 54

# 建立 card -> index 映射（字典）
CARD_TO_IDX = {c: i for i, c in enumerate(CARD_ORDER)}

# -------------------------
# 辅助函数：将 cards list 映射为 54-dim 向量
# -------------------------
def convert_cards_to_54(cards: List[str]) -> np.ndarray:
    """
    把 cards（例如 ['H2','D3','SR', ...]）转换为长度 54 的计数向量。
    计数范围 {0,1,2}（按你说明“每种牌最多两张”）
    返回 dtype=np.int8 或 np.float32（默认 float32）
    """
    vec = np.zeros(54, dtype=np.int8)
    if cards is None:
        return vec
    for c in cards:
        if c is None:
            continue
        if not isinstance(c, str):
            # 如果服务器给别的编码（如数字），你需要在调用前把它映射成 'H2' 这种字符串。
            continue
        idx = CARD_TO_IDX.get(c)
        if idx is None:
            # 不认识的牌字符串，忽略（也可以 raise）
            continue
        # cap at 2
        if vec[idx] < 2:
            vec[idx] += 1
    return vec

# -------------------------
# 解析 PlayCard 中的 'cards' 字段（你在说明里有三段结构）
# -------------------------
def parse_playcard_field(cards_field: Any) -> Dict[str, Any]:
    """
    根据你提供的说明，PlayCard 的 cards 字段分三部分:
      - 第一个元素: 出牌类型 (字符串，如 'single','pair','triple','bomb','straight', etc.)
      - 第二个元素: 点数/代表的主值 (例如 '2', 'A', 'T' 或最小牌的点数)
      - 第三部分: 具体花色字符串列表（例如 ['H2','D2']） 或更复杂结构

    返回 dict:
      {
        'type': str or None,
        'main_rank': str or None,
        'cards_list': list of card strings (具体牌，可能为空)
      }
    注意：按你实际 server schema 调整解析逻辑。
    """
    res = {'type': None, 'main_rank': None, 'cards_list': []}
    if not cards_field:
        return res
    # 若 cards_field 是列表且长度 >=1
    if isinstance(cards_field, list):
        if len(cards_field) >= 1:
            res['type'] = cards_field[0]
        if len(cards_field) >= 2:
            res['main_rank'] = cards_field[1]
        if len(cards_field) >= 3:
            # 第三项可能是 list （花色/具体牌），也可能是 str（逗号分隔）
            third = cards_field[2]
            if isinstance(third, list):
                res['cards_list'] = third
            elif isinstance(third, str):
                # 假设以空格或逗号分隔
                parts = third.replace(',', ' ').split()
                res['cards_list'] = parts
            else:
                # 其他格式，直接跳过
                res['cards_list'] = []
    else:
        # 如果不是 list，尽可能解析
        res['cards_list'] = []
    return res

# -------------------------
# 主函数: message_to_observation
# -------------------------
def message_to_observation(
    message: Dict[str, Any],
    K_history: int = 8,
    max_actions: int = 64,
    target_dim: int = 513
) -> Dict[str, Any]:
    """
    把来自 WebSocket 的 message(dict) 转换为结构化 observation 以及 固定长向量（可用于 NN）。

    参数:
    - message: dict, 已由 json.loads(message_str) 得到，至少包含:
        - 'operation' (str): e.g. "Deal","RequestAction","PlayCard","GameResult"
        - 'data' (dict): 消息具体内容（你之前给的示例里是 msg_data）
      注意：对你的具体服务器字段名若有不同，请在调用前把 message 转换成该约定结构，
            或在本函数里适配 field 名。
    - K_history: int, 最近 K 步 public history 用来编码（默认 8）
    - max_actions: int, 用于构造固定长度的 legal action mask（默认为 64）
    - target_dim: int, 最终扁平向量的目标维度（如果小于实际拼接长度会被截断；若更大会以 0 填充）

    返回:
    {
      'obs': {
          'hand_54': np.array(shape=(54,), dtype=int),
          'public_history': np.array(shape=(K_history,54), dtype=int),
          'last_play_54': np.array(shape=(54,), dtype=int),
          'last_play_type': str or None,
          'last_play_main_rank': str or None,
          'remaining_counts': np.array(shape=(4,), dtype=int),
          'position': int or None,
          'legal_actions': list or None,           # 原始 candidate actions 列表（若有）
          'legal_mask': np.array(shape=(max_actions,), dtype=bool) # padded mask
      },
      'obs_vector': np.array(shape=(target_dim,), dtype=float),
      'meta': { 'raw_operation': operation, 'raw_message': message }  # debug info
    }

    实现细节:
    - hand_54: 对 message['data']['cards'] 或 message['data']['hand_cards'] 做 convert_cards_to_54
    - public_history: 从 message['data'].get('recent_plays', []) 中取最近 K_history 步，每步转 54 向量
    - last_play_*: 若当前 message 是 PlayCard 含有出牌信息，则填充
    - remaining_counts: 若 message 中有 players remaining counts 填充，否则填 0
    - legal_mask: 若 message['data'] 中包含 'actions' 列表，则 mask[:len(actions)]=True
    - obs_vector: 拼接顺序 (并可被 target_dim 截断或 pad):
         concat = [ hand_54 (54),
                    public_history flattened (K_history*54),
                    last_play_54 (54),
                    remaining_counts (4),
                    position one-hot (4),
                    last_play_type one-hot (we use small dict -> d_type dims),
                    legal_mask as float (max_actions) ]
      以上只是默认顺序；函数会把 concat 后的向量做 zero-pad 或截断成 target_dim。
    """
    operation = message.get('operation') or message.get('op') or message.get('type')
    data = message.get('data', message)  # 有时 message 本身就是 data

    # 1) hand
    # 在 Deal / RequestAction 时 'cards' 通常是手牌数组（27 items），我们把它映射到 54-vector
    hand_cards = data.get('cards') or data.get('hand_cards') or []
    hand_54 = convert_cards_to_54(hand_cards)

    # 2) public_history: 最近 K 步 played cards (每步是 list of card strings)
    recent_plays = data.get('recent_plays') or data.get('public_history') or []
    # recent_plays 期望为 list of plays, 每个 play 为 list of card strings
    public_history = np.zeros((K_history, 54), dtype=np.int8)
    for i in range(min(K_history, len(recent_plays))):
        play = recent_plays[i]
        if isinstance(play, list):
            public_history[i] = convert_cards_to_54(play)
        else:
            # 如果 play 是复杂结构（如 dict），尝试提取 play['cards']
            if isinstance(play, dict) and 'cards' in play:
                public_history[i] = convert_cards_to_54(play['cards'])

    # 3) last play (如果 PlayCard 操作)
    last_play_54 = np.zeros(54, dtype=np.int8)
    last_play_type = None
    last_play_main_rank = None
    if operation == 'PlayCard' or data.get('operation') == 'PlayCard':
        # 可能字段是 data['cards'] 按你提供的格式（三段）
        parsed = parse_playcard_field(data.get('cards'))
        last_play_type = parsed.get('type')
        last_play_main_rank = parsed.get('main_rank')
        last_cards_list = parsed.get('cards_list', [])
        last_play_54 = convert_cards_to_54(last_cards_list)

    # 4) remaining_counts / position
    remaining_counts = np.array(data.get('remaining_counts') or data.get('remains') or [0,0,0,0], dtype=np.int8)
    # ensure shape (4,)
    if remaining_counts.shape[0] < 4:
        tmp = np.zeros(4, dtype=np.int8)
        tmp[:remaining_counts.shape[0]] = remaining_counts
        remaining_counts = tmp
    elif remaining_counts.shape[0] > 4:
        remaining_counts = remaining_counts[:4]

    position = data.get('position', None)  # 服务器分配的位置 (0..3) 或 None

    # 5) legal actions (RequestAction)
    candidate_actions = data.get('actions', None)  # 服务器给的候选动作列表（原始）
    legal_mask = np.zeros(max_actions, dtype=np.bool_)
    if candidate_actions is not None:
        n = len(candidate_actions)
        legal_mask[:min(n, max_actions)] = True

    # 6) last_play_type one-hot encoding (we define a small mapping,扩展可在外部传入)
    # 预定义常见动作类型（按论文/实现可改）
    ACTION_TYPE_LIST = ['single','pair','triple','three_plus_two','straight','bomb','pass','other']
    type_onehot = np.zeros(len(ACTION_TYPE_LIST), dtype=np.int8)
    if last_play_type:
        try:
            idx = ACTION_TYPE_LIST.index(str(last_play_type))
            type_onehot[idx] = 1
        except ValueError:
            # 未知类型 -> 'other'
            type_onehot[-1] = 1

    # 7) position one-hot (4)
    pos_onehot = np.zeros(4, dtype=np.int8)
    if position is not None and isinstance(position, int) and 0 <= position < 4:
        pos_onehot[position] = 1

    # 8) 构建结构化 obs dict
    obs_struct = {
        'hand_54': hand_54,                       # shape (54,)
        'public_history': public_history,         # shape (K_history,54)
        'last_play_54': last_play_54,             # shape (54,)
        'last_play_type': last_play_type,         # 原始字符串
        'last_play_main_rank': last_play_main_rank,
        'remaining_counts': remaining_counts,     # shape (4,)
        'position': position,
        'position_onehot': pos_onehot,            # shape (4,)
        'last_play_type_onehot': type_onehot,     # shape (len(ACTION_TYPE_LIST),)
        'legal_actions': candidate_actions,       # 原始动作列表
        'legal_mask': legal_mask                  # shape (max_actions,)
    }

    # 9) 拼接成扁平向量（按固定顺序），然后 pad/truncate 到 target_dim
    #  拼接顺序（可按需改）:
    #    [ hand_54,
    #      public_history.flatten() (K_history*54),
    #      last_play_54,
    #      remaining_counts (4),
    #      position_onehot (4),
    #      last_play_type_onehot (T),
    #      legal_mask (max_actions as float 0/1)
    #    ]
    components = []
    components.append(obs_struct['hand_54'].astype(np.float32).reshape(-1))
    components.append(obs_struct['public_history'].astype(np.float32).reshape(-1))
    components.append(obs_struct['last_play_54'].astype(np.float32).reshape(-1))
    components.append(obs_struct['remaining_counts'].astype(np.float32).reshape(-1))
    components.append(obs_struct['position_onehot'].astype(np.float32).reshape(-1))
    components.append(obs_struct['last_play_type_onehot'].astype(np.float32).reshape(-1))
    components.append(obs_struct['legal_mask'].astype(np.float32).reshape(-1))

    flat = np.concatenate(components, axis=0) if len(components) > 0 else np.zeros(0, dtype=np.float32)

    # pad or truncate to target_dim
    if target_dim is None:
        obs_vector = flat.astype(np.float32)
    else:
        if flat.size >= target_dim:
            obs_vector = flat[:target_dim].astype(np.float32)
        else:
            pad = np.zeros((target_dim - flat.size,), dtype=np.float32)
            obs_vector = np.concatenate([flat.astype(np.float32), pad], axis=0)

    # output
    return {
        'obs': obs_struct,
        'obs_vector': obs_vector,
        'meta': {
            'raw_operation': operation,
            'raw_message': message
        }
    }

# -------------------------
# 小测试（示例）: 你应该把真实 message 换进去试验
# -------------------------
if __name__ == "__main__":
    # 示例 Deal 消息
    deal_msg = {
        "operation": "Deal",
        "data": {
            "cards": ["H2","C2","S2","D2","H3","HB"],  # 示例
            "position": 1,
            "remaining_counts": [27, 27, 27, 27]
        }
    }
    out = message_to_observation(deal_msg, K_history=4, max_actions=16, target_dim=513)
    print("obs_vector shape:", out['obs_vector'].shape)
    print("first 60 dims:", out['obs_vector'])
    # 示例 RequestAction 消息
    req_msg = {
        "operation": "RequestAction",
        "data": {
            "cards": ["H2","C2","S2"],
            "recent_plays": [["H4","D4"], ["H5"]],
            "remaining_counts": [10,12,8,7],
            "position": 0,
            "actions": ["pass","play_single_H2","play_pair_2"]
        }
    }
    out2 = message_to_observation(req_msg, K_history=4, max_actions=8, target_dim=513)
    print("legal_mask sum:", out2['obs']['legal_mask'].sum(), " vector sample:", out2['obs_vector'][:80])

