# ppo_ws_actor.py
import asyncio
import json
import logging
import numpy as np
import websockets
from ppo_learner import PPOLearner  # 假设和之前 mock 一致
from message2state import convert_message_to_state  # 你之前写的 convert_message_to_state

logging.basicConfig(level=logging.INFO)

class PPOWSActor:
    def __init__(self, key, learner_agent):
        self.key = key
        self.uri = f"ws://localhost:23456/{key}"  # 修改为真实服务器
        self.cards = []
        self.position = None
        self.round_num = 0
        self.agent = learner_agent
        self.ws = None

    async def connect(self):
        try:
            self.ws = await websockets.connect(self.uri)
            logging.info(f"[{self.key}] 连接成功")
        except Exception as e:
            logging.error(f"[{self.key}] 连接失败: {e}")
            raise e

    async def handle_message(self, message):
        data = json.loads(message)
        op = data.get("operation")
        msg_data = data.get("data", {})

        if op == "Deal":
            self.cards = msg_data.get("cards", [])
            self.position = msg_data.get("position", 0)
            logging.info(f"[{self.key}] 第{self.round_num+1}轮收到手牌: {self.cards}")

        elif op == "RequestAction":
            actions = msg_data.get("actions", [])
            state, mask = convert_message_to_state(self.cards, actions, self.round_num)
            action_index = self.agent.select_action(state, mask)
            # 发送动作给服务器
            response = {"operation": "Action", "actionIndex": int(action_index)}
            await self.ws.send(json.dumps(response))
            logging.info(f"[{self.key}] 出牌动作: {action_index}")

        elif op == "PlayCard":
            # 记录其他玩家动作，如果想用在状态构建可加历史
            pass

        elif op == "GameResult":
            reward = msg_data.get("rank", 0)  # 可以改成你设定的奖励策略
            done = True
            self.agent.add_experience(
                state=np.zeros_like(mask),  # 如果你想用最后状态，可修改
                action=0,
                reward=reward,
                next_state=np.zeros_like(mask),
                done=done,
                mask=mask
            )
            self.round_num += 1
            self.cards = []

    async def run(self):
        await self.connect()
        async for msg in self.ws:
            await self.handle_message(msg)

# ------------------------
# 运行多个 Actor
# ------------------------
async def main():
    from ppo_learner import PPOLearner
    # 假设 learner 已经初始化好了
    learner_agent = PPOLearner(state_dim=55, max_actions=50)

    keys = ["a1", "b1", "a2", "b2"]  # 你的玩家 key
    actors = [PPOWSActor(k, learner_agent) for k in keys]

    await asyncio.gather(*[actor.run() for actor in actors])

if __name__ == "__main__":
    asyncio.run(main())
