import sys
import json
import asyncio
import websockets
import logging
import random
from typing import List, Dict, Any
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QGridLayout, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont

def setup_logger(key):
    """为每个客户端创建独立的日志配置"""
    logger = logging.getLogger(f'gd_client_{key}')
    
    # 如果logger已经有处理器，说明已经配置过，直接返回
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器，使用key作为文件名
    file_handler = logging.FileHandler(f'gd_client_{key}.log')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class WebSocketThread(QThread):
    """WebSocket通信线程"""
    message_received = pyqtSignal(str)
    connection_closed = pyqtSignal()
    
    def __init__(self, key):
        super().__init__()
        self.key = key
        self.uri = f"ws://localhost:23456/{key}"  # 固定使用localhost
        self.ws = None
        self.running = True
        # 使用key初始化独立的日志记录器
        self.logger = setup_logger(key)
        
    def run(self):
        asyncio.run(self._run())
        
    async def _run(self):
        try:
            self.ws = await websockets.connect(self.uri)
            self.logger.info(f"玩家{self.key}连接成功")
            
            while self.running:
                try:
                    message = await self.ws.recv()
                    self.message_received.emit(message)
                except websockets.exceptions.ConnectionClosed:
                    self.logger.info("连接已关闭")
                    self.connection_closed.emit()
                    break
        except Exception as e:
            self.logger.error(f"WebSocket错误: {e}")
            self.connection_closed.emit()
            
    def stop(self):
        self.running = False
        if self.ws:
            asyncio.run(self.ws.close())

class CardWidget(QWidget):
    """单张牌的显示组件"""
    def __init__(self, card_value, parent=None):
        super().__init__(parent)
        self.card_value = card_value
        self.setFixedSize(60, 90)
        
    def to_local_card(self, card_str):
        """将字符串格式的牌转换为本地格式"""
        if not card_str:
            return 0
            
        # 如果输入是列表格式（如['Single', '6', ['H6']]），取最后一个元素
        if isinstance(card_str, list):
            if len(card_str) == 3 and isinstance(card_str[2], list):
                card_str = card_str[2][0]  # 取['H6']中的'H6'
            else:
                return 0
            
        # 解析花色和点数
        suit = card_str[0].upper()
        rank = card_str[1:].upper()
        
        # 转换花色
        suit_val = 0
        if suit == 'H':  # 红桃
            suit_val = 1
        elif suit == 'D':  # 方块
            suit_val = 2
        elif suit == 'C':  # 梅花
            suit_val = 3
        elif suit == 'S':  # 黑桃
            suit_val = 4
            
        # 转换点数
        rank_val = 0
        if rank == 'B':  # 小王
            rank_val = 16
        elif rank == 'R':  # 大王
            rank_val = 17
        elif rank == 'A':
            rank_val = 14
        elif rank == 'K':
            rank_val = 13
        elif rank == 'Q':
            rank_val = 12
        elif rank == 'J':
            rank_val = 11
        elif rank == 'T':
            rank_val = 10
        else:
            try:
                rank_val = int(rank)
            except ValueError:
                rank_val = 0
                
        # 组合花色和点数
        return (suit_val << 5) | rank_val
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制牌背景
        painter.fillRect(0, 0, self.width(), self.height(), QColor(255, 255, 255))
        
        # 绘制牌边框
        painter.setPen(QColor(0, 0, 0))
        painter.drawRect(0, 0, self.width()-1, self.height()-1)
        
        # 将字符串格式的牌转换为本地格式
        card_val = self.to_local_card(self.card_value)
        
        # 解析牌值
        suit = (card_val & 0xE0) >> 5  # 花色
        rank = card_val & 0x1F         # 点数
        
        # 设置花色颜色
        if suit in [1, 2]:  # 红桃和方块为红色
            painter.setPen(QColor(255, 0, 0))
        else:  # 梅花和黑桃为黑色
            painter.setPen(QColor(0, 0, 0))
            
        # 绘制花色符号
        suit_symbol = ""
        if suit == 1:  # 红桃
            suit_symbol = "♥"
        elif suit == 2:  # 方块
            suit_symbol = "♦"
        elif suit == 3:  # 梅花
            suit_symbol = "♣"
        elif suit == 4:  # 黑桃
            suit_symbol = "♠"
            
        # 绘制点数
        rank_str = ""
        if rank == 16:  # 小王
            rank_str = "小王"
        elif rank == 17:  # 大王
            rank_str = "大王"
        elif rank == 14:  # A
            rank_str = "A"
        elif rank == 13:  # K
            rank_str = "K"
        elif rank == 12:  # Q
            rank_str = "Q"
        elif rank == 11:  # J
            rank_str = "J"
        elif rank == 10:  # 10
            rank_str = "10"
        else:
            rank_str = str(rank)
            
        # 设置字体
        font = QFont("Arial", 12)
        painter.setFont(font)
        
        # 绘制左上角的点数和花色
        if rank_str in ["小王", "大王"]:
            painter.drawText(5, 20, rank_str)
        else:
            painter.drawText(5, 20, rank_str)
            painter.drawText(5, 40, suit_symbol)
            
        # 绘制右下角的点数和花色（倒置）
        if rank_str not in ["小王", "大王"]:
            painter.save()
            painter.translate(self.width(), self.height())
            painter.rotate(180)
            painter.drawText(-55, -20, rank_str)
            painter.drawText(-55, -40, suit_symbol)
            painter.restore()

class PlayerArea(QWidget):
    """玩家区域，显示剩余牌数和出牌"""
    def __init__(self, position, parent=None):
        super().__init__(parent)
        self.position = position
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        
        # 玩家标签和剩余牌数合并为一个标签
        self.info_label = QLabel(f"玩家{position + 1} (剩余: 0)")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.info_label)
        
        # 出牌区域（移到上面）
        self.play_cards_area = QWidget()
        self.play_cards_layout = QHBoxLayout()
        self.play_cards_layout.setSpacing(-40)
        self.play_cards_area.setLayout(self.play_cards_layout)
        self.layout.addWidget(self.play_cards_area)
        
        # 手牌区域（移到下面）
        self.hand_cards_area = QWidget()
        self.hand_cards_layout = QHBoxLayout()
        self.hand_cards_layout.setSpacing(-40)
        self.hand_cards_area.setLayout(self.hand_cards_layout)
        self.layout.addWidget(self.hand_cards_area)
        
    def set_remaining_cards(self, count):
        """设置剩余牌数"""
        self.info_label.setText(f"玩家{self.position + 1} (剩余: {count})")
        
    def clear_hand_cards(self):
        """清除所有手牌"""
        while self.hand_cards_layout.count():
            item = self.hand_cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
    def set_hand_cards(self, cards):
        """设置手牌"""
        self.clear_hand_cards()
        if cards:
            card_container = QWidget()
            card_layout = QHBoxLayout()
            card_layout.setSpacing(-40)
            card_container.setLayout(card_layout)
            
            for card_value in cards:
                card = CardWidget(card_value)
                card_layout.addWidget(card)
            
            self.hand_cards_layout.addWidget(card_container)
        
    def clear_play_cards(self):
        """清除所有出牌"""
        while self.play_cards_layout.count():
            item = self.play_cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
    def set_play_cards(self, cards, is_pass=False):
        """设置出牌"""
        self.clear_play_cards()
        # 判断pass的多种情况
        is_pass_flag = is_pass
        if not is_pass_flag:
            # 1. 服务器直接传字符串 'pass'
            if isinstance(cards, str) and cards.lower() == 'pass':
                is_pass_flag = True
            # 2. 服务器传 ['PASS', 'PASS', 'PASS'] 或类似全为'PASS'的列表
            elif isinstance(cards, list) and all(str(c).upper() == 'PASS' for c in cards):
                is_pass_flag = True
        if is_pass_flag:
            pass_label = QLabel("不出")
            pass_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pass_label.setStyleSheet("color: red; font-weight: bold;")
            self.play_cards_layout.addWidget(pass_label)
        elif cards:
            # 处理服务器发来的 ["Bomb","5",["H5","S5","C5","D5"]] 结构
            if isinstance(cards, list) and len(cards) == 3 and isinstance(cards[2], list):
                card_list = cards[2]
            else:
                card_list = cards if isinstance(cards, list) else [cards]
            card_container = QWidget()
            card_layout = QHBoxLayout()
            card_layout.setSpacing(-40)
            card_container.setLayout(card_layout)
            for card_value in card_list:
                card = CardWidget(card_value)
                card_layout.addWidget(card)
            self.play_cards_layout.addWidget(card_container)

class GameArea(QWidget):
    """游戏区域，显示四个玩家的牌"""
    def __init__(self, external_position, parent=None):
        super().__init__(parent)
        self.external_position = external_position  # 外部玩家位置
        self.layout = QGridLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)
        
        # 创建四个玩家区域
        self.player_areas = {
            0: PlayerArea(0),  # 上家
            1: PlayerArea(1),  # 右家
            2: PlayerArea(2),  # 下家
            3: PlayerArea(3)   # 左家
        }
        
        # 根据外部玩家位置设置布局
        # 计算其他玩家的相对位置
        left_pos = (external_position + 3) % 4   # 左家
        top_pos = (external_position + 2) % 4    # 上家
        right_pos = (external_position + 1) % 4  # 右家
        
        # 设置布局
        self.layout.addWidget(self.player_areas[top_pos], 0, 1, 1, 1, Qt.AlignmentFlag.AlignBottom)    # 上家
        self.layout.addWidget(self.player_areas[right_pos], 1, 2, 1, 1, Qt.AlignmentFlag.AlignLeft)    # 右家
        self.layout.addWidget(self.player_areas[external_position], 2, 1, 1, 1, Qt.AlignmentFlag.AlignTop)  # 下家（外部玩家）
        self.layout.addWidget(self.player_areas[left_pos], 1, 0, 1, 1, Qt.AlignmentFlag.AlignRight)    # 左家
        
        # 设置行列的拉伸因子
        self.layout.setRowStretch(0, 1)
        self.layout.setRowStretch(1, 1)
        self.layout.setRowStretch(2, 1)
        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)
        self.layout.setColumnStretch(2, 1)
        
    def update_remaining_cards(self, position, count):
        """更新指定位置的剩余牌数"""
        self.player_areas[position].set_remaining_cards(count)
        
    def clear_all_play_cards(self):
        """清除所有玩家的出牌"""
        for area in self.player_areas.values():
            area.clear_play_cards()
        
    def update_play_cards(self, position, cards, is_pass=False):
        """更新指定位置的出牌"""
        self.player_areas[position].set_play_cards(cards, is_pass)
        
    def update_hand_cards(self, position, cards):
        """更新指定位置的手牌"""
        self.player_areas[position].set_hand_cards(cards)

class GameInfoArea(QWidget):
    """游戏信息区域"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(10, 0, 10, 0)
        self.setLayout(self.layout)
        
        # 添加信息标签
        self.round_label = QLabel("当前轮次: 0")
        self.status_label = QLabel("游戏状态: 等待开始")
        self.rank_label = QLabel("当前级牌: 2")  # 新增级牌显示
        
        # 将标签添加到水平布局
        self.layout.addWidget(self.round_label)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.rank_label)  # 添加级牌标签
        
    def update_info(self, round_num, status, rank=None):
        """更新游戏信息"""
        self.round_label.setText(f"当前轮次: {round_num}")
        self.status_label.setText(f"游戏状态: {status}")
        if rank is not None:
            self.rank_label.setText(f"当前级牌: {rank}")

class GDTestClient(QMainWindow):
    def __init__(self, key: str):
        super().__init__()
        self.key = key
        self.uri = f"ws://localhost:23456/{key}"  # 固定使用localhost
        self.cards: List[int] = []
        self.ws = None
        self.current_round = 0
        self.total_rounds = 0
        self.game_stats = []
        
        # 从key中获取位置信息（a1->0, b1->1, a2->2, b2->3）
        self.position = 0
        if key.startswith('a'):
            self.position = 0 if key.endswith('1') else 2
        else:  # b
            self.position = 1 if key.endswith('1') else 3
        
        # 使用key初始化独立的日志记录器
        self.logger = setup_logger(key)
        
        # 设置窗口
        self.setWindowTitle(f"掼蛋游戏客户端 - 玩家{key}(位置{self.position})")
        self.setMinimumSize(800, 600)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        # 创建游戏区域，传入外部玩家位置
        self.game_area = GameArea(self.position)
        main_layout.addWidget(self.game_area)
        
        # 创建游戏信息区域
        self.game_info_area = GameInfoArea()
        main_layout.addWidget(self.game_info_area, alignment=Qt.AlignmentFlag.AlignRight)
        
        # 启动WebSocket线程
        self.ws_thread = WebSocketThread(key)
        self.ws_thread.message_received.connect(self.handle_message)
        self.ws_thread.connection_closed.connect(self.on_connection_closed)
        self.ws_thread.start()

    def on_connection_closed(self):
        """处理连接关闭"""
        self.statusBar().showMessage("连接已关闭")
        
    def closeEvent(self, event):
        """窗口关闭时停止WebSocket线程"""
        self.ws_thread.stop()
        event.accept()

    async def handle_action_request(self, actions):
        """处理出牌请求"""
        if not actions:  # 如果 actions 是空的
            action_index = 0  # 表示不出牌
        else:
            # 随机选择一个动作，action为dict，取index字段
            action_index = random.choice(actions)["index"]

        # 构造响应
        response = {
            "operation": "Action",
            "actionIndex": action_index
        }

        # 发送响应
        await self.ws_thread.ws.send(json.dumps(response))

    def show_game_result(self, win_list, my_rank, time_used):
        """显示游戏结果弹框"""
        # 构建结果消息
        result_msg = f"游戏结束！\n\n"
        result_msg += f"级牌: {my_rank}\n"
        result_msg += f"用时: {time_used}秒\n\n"
        result_msg += "完整排名:\n"
        for i, rank in enumerate(win_list):
            result_msg += f"玩家{i + 1}: 第{rank}名\n"
        
        # 显示消息框
        QMessageBox.information(self, "游戏结果", result_msg)

    def show_game_statistics(self, stats_data):
        """显示游戏统计信息弹框"""
        # 构建统计信息消息
        stats_msg = "游戏统计信息：\n\n"
        stats_msg += f"总轮数: {stats_data['totalRounds']}\n"
        stats_msg += f"完成轮数: {stats_data['completedRounds']}\n"
        stats_msg += f"失败轮数: {stats_data['failedRounds']}\n"
        
        # 添加队伍得分统计
        stats_msg += "\n队伍得分统计：\n"
        # 队伍1（玩家1和3）
        team1_score = stats_data['playerWinCounts'].get('1', 0)
        stats_msg += f"队伍1（玩家1和3）: {team1_score}分\n"
        # 队伍2（玩家2和4）
        team2_score = stats_data['playerWinCounts'].get('2', 0)
        stats_msg += f"队伍2（玩家2和4）: {team2_score}分\n"
        
        # 添加玩家得分明细
        stats_msg += "\n玩家得分明细：\n"
        for player, score in sorted(stats_data['playerWinCounts'].items()):
            stats_msg += f"玩家{player}: {score}分\n"
        
        # 显示消息框
        QMessageBox.information(self, "游戏统计", stats_msg)

    def handle_message(self, message: str):
        """
        处理收到的消息
        :param message: WebSocket消息
        """
        try:
            data = json.loads(message)
            operation = data.get("operation")
            phase = data.get("phase")
            msg_data = data.get("data", {})

            self.logger.info(f"收到消息: {message}")

            if operation == "Deal":
                # 处理发牌消息，更新剩余牌数
                position = msg_data.get("position")
                cards = msg_data.get("cards", [])
                self.game_area.update_remaining_cards(position, len(cards))
                # 更新游戏状态
                self.game_info_area.update_info(self.current_round, "游戏中")
                # 清除上一轮的出牌
                self.game_area.clear_all_play_cards()

            elif operation == "RequestAction":
                # 处理出牌请求
                actions = msg_data.get("actions", [])
                cards = msg_data.get("cards", [])  # 获取手牌
                position = msg_data.get("position")  # 获取当前玩家位置
                rank = msg_data.get("rank", 2)  # 获取级牌
                
                # 显示手牌
                self.game_area.update_hand_cards(position, cards)
                # 更新级牌显示
                self.game_info_area.update_info(self.current_round, "游戏中", rank)
                
                asyncio.run(self.handle_action_request(actions))

            elif operation == "PlayCard":
                # 处理其他玩家的出牌信息
                play_position = msg_data.get("position")
                cards = msg_data.get("cards", [])
                remaining_cards = msg_data.get("remainingCards", 0)
                action = msg_data.get("action", "")
                rank = msg_data.get("rank", 2)  # 获取级牌
                
                # 更新剩余牌数
                self.game_area.update_remaining_cards(play_position, remaining_cards)
                self.game_info_area.update_info(self.current_round, "游戏中", rank)
                
                # 显示所有玩家的出牌，包括外部用户
                if action == "pass":
                    self.logger.info(f"第{self.current_round + 1}轮 - 玩家{play_position + 1}选择不出")
                    # 显示pass
                    self.game_area.update_play_cards(play_position, [], is_pass=True)
                else:
                    self.logger.info(f"第{self.current_round + 1}轮 - 玩家{play_position + 1}出牌: {cards}")
                    # 更新出牌显示
                    self.game_area.update_play_cards(play_position, cards)

            elif operation == "GameResult":
                # 处理游戏结果
                self.current_round += 1
                round_result = msg_data.get("winList", [])
                my_rank = msg_data.get("rank", 0)
                time_used = msg_data.get("time", 0)
                
                # 记录本轮统计信息
                self.game_stats.append({
                    "round": self.current_round,
                    "rank": my_rank,
                    "results": round_result,
                    "time": time_used
                })
                
                self.logger.info(f"第{self.current_round}轮结束")
                self.logger.info(f"级牌: {my_rank}")
                self.logger.info(f"完整排名: {round_result}")
                self.logger.info(f"用时: {time_used}秒")
                
                # 更新GUI显示
                self.game_info_area.update_info(self.current_round, "游戏结束")
                
                # 显示游戏结果弹框
                self.show_game_result(round_result, my_rank, time_used)
                
                # 准备开始新的一轮
                self.cards = []
                # 清除所有出牌
                self.game_area.clear_all_play_cards()

            elif operation == "GameStatistics":
                # 处理游戏统计信息
                self.show_game_statistics(msg_data)
                # 更新游戏状态
                self.game_info_area.update_info(self.current_round, "游戏结束")

        except json.JSONDecodeError:
            self.logger.error(f"消息解析失败: {message}")
        except Exception as e:
            self.logger.error(f"处理消息时出错: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 获取玩家key参数
    if len(sys.argv) > 1:
        key = sys.argv[1]
    else:
        print("错误：请提供玩家唯一key参数(如a1、b1、a2、b2)")
        sys.exit(1)
    window = GDTestClient(key)
    window.show()
    sys.exit(app.exec()) 