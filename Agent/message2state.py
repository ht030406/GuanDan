import numpy as np

CARD_ORDER = ['H2', 'C2', 'S2', 'D2', 'H3', 'C3', 'S3', 'D3', 'H4', 'C4', 'S4', 'D4',
              'H5', 'C5', 'S5', 'D5', 'H6', 'C6', 'S6', 'D6', 'H7', 'C7', 'S7', 'D7',
              'H8', 'C8', 'S8', 'D8', 'H9', 'C9', 'S9', 'D9', 'HT', 'CT', 'ST', 'DT',
              'HJ', 'CJ', 'SJ', 'DJ', 'HQ', 'CQ', 'SQ', 'DQ', 'HK', 'CK', 'SK', 'DK',
              'HA', 'CA', 'SA', 'DA', 'HB', 'SR']

# def convert_message_to_state(actions, origin_cards, played_cards, up_player_played, teammate_played, others_played1
#                              , others_played2, others_played3, remaining_counts_others, wild_cards, round_num=0, max_cards=54, max_actions=50):
#     """
#     转换消息为固定维度的状态向量
#     :param cards: 自己手牌 list[int] or list[str]
#     :param actions: 当前可选动作列表，每个动作是 dict，至少包含 "index"
#     :param round_num: 当前轮次
#     :param max_cards: 最大牌数（掼蛋54张）
#     :param max_actions: 最大候选动作数
#     :return: state向量 (numpy array), action_mask
#     """
#
#     # --- 动作 mask ---
#     action_mask = np.zeros(max_actions, dtype=np.float32)
#     for a in actions:
#         try:
#             idx = int(a["index"])  # 这里强制转 int
#         except Exception:
#             continue
#         if 0 <= idx < max_actions:
#             action_mask[idx] = 1
#
#     # --- 轮次简单归一化 ---
#     round_feature = np.array([float(round_num) / 100.0], dtype=np.float32)
#
#     # --- 拼接状态 ---
#     state = np.concatenate([card_vec, round_feature])
#
#     return state, action_mask


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


def convert_message_to_state(actions,origin_cards, played_cards, up_player_played, teammate_played, others_played1
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


    # --- 动作 mask ---
    action_mask = np.zeros(5000, dtype=np.float32)
    for a in actions:
        try:
            idx = int(a["index"])  # 这里强制转 int
        except Exception:
            continue
        if 0 <= idx < 5000:
            action_mask[idx] = 1

    return state,action_mask
