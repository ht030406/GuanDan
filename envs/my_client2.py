import asyncio
import websockets
import json
import random
import logging
import sys
import os
from typing import List, Dict, Any
from Agent.agent import SimpleAgent
from Agent.message2state import convert_message_to_state    #调用状态转换函数

def setup_logger(key):
    """为每个客户端创建独立的日志配置"""
    logger = logging.getLogger(f'test_client_{key}')
    
    # 如果logger已经有处理器，说明已经配置过，直接返回
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器，使用key作为文件名
    file_handler = logging.FileHandler(f'test_client_{key}.log')
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

class GDTestClient:
    def __init__(self, key: str, agent):
        """
        初始化测试客户端
        :param key: 玩家唯一key（如a1、b1、a2、b2）
        """
        self.key = key
        self.uri = f"ws://localhost:23456/{key}"
        self.cards: List[int] = []
        self.ws = None
        self.current_round = 0
        self.total_rounds = 0
        self.game_stats = []  # 记录每轮的统计信息
        self.position = None  # 位置信息将从服务器消息中获取
        # 使用key初始化独立的日志记录器
        self.logger = setup_logger(key)
        self.agent = agent

    async def connect(self):
        """建立WebSocket连接"""
        try:
            self.ws = await websockets.connect(self.uri)
            self.logger.info(f"玩家{self.key}连接成功")
            return True
        except Exception as e:
            self.logger.error(f"连接失败: {e}")
            return False

    async def handle_message(self, message: str):
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
                # 处理发牌消息
                self.cards = msg_data.get("cards", [])
                # 从消息中获取位置信息
                self.position = msg_data.get("position")
                self.logger.info(f"第{self.current_round + 1}轮 - 收到牌: {self.cards}")

            elif operation == "RequestAction":
                # 处理出牌请求
                actions = msg_data.get("actions", [])
                await self.handle_action_request(actions)

            elif operation == "PlayCard":
                # 处理其他玩家的出牌信息
                play_position = msg_data.get("position")
                cards = msg_data.get("cards", [])
                action = msg_data.get("action", "")
                if play_position != self.key:
                    if action == "pass":
                        self.logger.info(f"第{self.current_round + 1}轮 - 玩家{play_position + 1}选择不出")
                    else:
                        self.logger.info(f"第{self.current_round + 1}轮 - 玩家{play_position + 1}出牌: {cards}")

            elif operation == "GameResult":
                # 处理游戏结果
                self.current_round += 1
                round_result = msg_data.get("winList", [])
                my_rank = msg_data.get("rank", 0)
                time_used = msg_data.get("time", 0)
                
                # 记录本轮统计信息
                if self.position is not None:
                    self.game_stats.append({
                        "round": self.current_round,
                        "player_rank": round_result[self.position],  # 使用服务器分配的位置
                        "results": round_result,
                        "time": time_used
                    })
                
                self.logger.info(f"第{self.current_round}轮结束")
                self.logger.info(f"级牌: {my_rank}")
                self.logger.info(f"完整排名: {round_result}")
                self.logger.info(f"用时: {time_used}秒")
                
                # 准备开始新的一轮
                self.cards = []

        except json.JSONDecodeError:
            self.logger.error(f"消息解析失败: {message}")
        except Exception as e:
            self.logger.error(f"处理消息时出错: {e}")

    async def handle_action_request(self, actions: List[Dict[str, Any]]):
        try:
            # 转换状态
            state, action_mask = convert_message_to_state(
                self.cards, actions, self.current_round
            )

            # agent选择动作
            action_index = self.agent(state, action_mask)

            self.logger.info(f"第{self.current_round + 1}轮 - Agent选择动作: {action_index}")

            response = {
                "operation": "Action",
                "actionIndex": int(action_index)
            }
            await self.ws.send(json.dumps(response))

        except Exception as e:
            self.logger.error(f"处理出牌请求时出错: {e}")

    async def run(self):
        """运行客户端"""
        if not await self.connect():
            return

        try:
            async for message in self.ws:
                await self.handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"连接已关闭，共完成 {self.current_round} 轮游戏")
        except Exception as e:
            self.logger.error(f"运行时出错: {e}")
        finally:
            if self.ws:
                await self.ws.close()

async def main():
    """
    主函数，创建并运行测试客户端
    使用示例：python my_client2.py a1
    """
    import argparse
    parser = argparse.ArgumentParser(description='掼蛋游戏测试客户端')
    parser.add_argument('key', type=str, help='玩家唯一key(如a1、b1、a2、b2)')
    args = parser.parse_args()

    agent = SimpleAgent(state_dim=55, max_actions=50)  # 54张牌+1轮次
    client = GDTestClient(args.key,agent)
    await client.run()

if __name__ == "__main__":
    asyncio.run(main()) 