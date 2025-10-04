# actor/actor.py
"""
Actor wrapper for GDTestClient that collects (s,a,r,s',done,mask,logp,value) transitions
and writes them into a ReplayBuffer (or POSTs to a learner HTTP endpoint).

Usage:
    python -m actor.actor --key a1 --mode local

Notes:
- Adjust the import path for GDTestClient and SimpleAgent if needed.
- This actor keeps a short list of pending actions (one per RequestAction) and when
  receives PlayResult/GameResult will complete pending and push to buffer.
"""

import asyncio
import argparse
import json
import time
import threading
from typing import List, Dict, Any, Optional
from envs.my_client1 import GDTestClient  # preferred location based on earlier plan

from Agent.agent import SimpleAgent
from storage.replay_buffer import ReplayBuffer
from Agent.message2state import convert_message_to_state  # your state conversion

# If your SimpleAgent is in another path, adjust import accordingly.

# -------------------------
# Actor class (subclass GDTestClient)
# -------------------------
class RLActorClient(GDTestClient):
    """
    Extend GDTestClient to collect transitions into a ReplayBuffer.
    It overrides handle_action_request and handle_message to record pending transitions
    and finalize them when rewards arrive.
    """
    def __init__(self, key: str, agent: SimpleAgent, buffer: ReplayBuffer,
                 learner_http_url: Optional[str] = None):
        super().__init__(key, agent)
        self.replay_buffer = buffer
        self.learner_http_url = learner_http_url  # optional remote upload endpoint
        # pending actions list: each entry is dict with keys:
        # {'obs': np.array, 'action': int, 'mask': np.array, 'logp': float or None, 'timestamp': float}
        self._pending_actions: List[Dict[str, Any]] = []
        self._pending_lock = threading.Lock()
        # configure how many pending we keep before flushing when no reward arrives
        self.pending_timeout = 10.0  # seconds

    # override: when server requests action, record the chosen action as pending
    async def handle_action_request(self, actions: List[Dict[str, Any]]):
        """
        This method replaces the parent method behavior for decision recording.
        We compute state & mask by calling convert_message_to_state (like your current code),
        call agent to choose action, store pending transition, and send action to server.
        """
        try:
            # Use the same state conversion as you already use
            state, action_mask = convert_message_to_state(
                actions,
                self.cards,
                self.played_cards,
                self.up_player_played,
                self.teammate_played,
                self.others_played1,
                self.others_played2,
                self.others_played3,
                self.remaining_counts_others,
                self.wild_cards
            )

            # get action from agent (SimpleAgent returns index)
            # If your agent later returns (idx, logp), adapt here.
            action_index = self.agent(state, action_mask)

            # send action to server (same as your previous behavior)
            response = {"operation": "Action", "actionIndex": int(action_index)}
            await self.ws.send(json.dumps(response))

            # record pending transition (we'll fill reward/next_obs/done later)
            pending = {
                "obs": state.astype(float).tolist() if hasattr(state, "astype") else state,
                "action": int(action_index),
                "mask": action_mask.tolist() if hasattr(action_mask, "tolist") else action_mask,
                "logp": None,
                "value": None,
                "timestamp": time.time()
            }
            with self._pending_lock:
                self._pending_actions.append(pending)

            # optional: log
            self.logger.info(f"[Actor] chosen action {action_index} and queued pending (pending_count={len(self._pending_actions)})")

        except Exception as e:
            self.logger.error(f"actor.handle_action_request error: {e}")

    # override or extend handle_message to finalize pending transitions when reward/info arrives
    async def handle_message(self, message: str):
        """
        We mostly reuse parent logic, but look for PlayResult / GameResult to finalize pending transitions.
        """
        try:
            data = json.loads(message)
            operation = data.get("operation")
            msg_data = data.get("data", {})

            # call parent handling for existing logging/state updates
            # (We use super().handle_message to keep existing behavior; it will call our handle_action_request override when needed)
            await super().handle_message(message)

            # now post-process certain messages
            if operation in ("RequestAction", "Reward"):
                # many servers send immediate reward info here
                actions = msg_data.get("actions", [])
                reward = float(msg_data.get("reward", 0.0))
                done = bool(msg_data.get("done", False))
                # next obs: we can convert current message to next state
                try:
                    next_state, next_mask = convert_message_to_state(
                        actions,
                        # Use current stored fields if needed; convert_message_to_state should be robust
                        self.cards,
                        self.played_cards,
                        self.up_player_played,
                        self.teammate_played,
                        self.others_played1,
                        self.others_played2,
                        self.others_played3,
                        self.remaining_counts_others,
                        self.wild_cards
                    )
                    next_obs_serial = next_state.astype(float).tolist() if hasattr(next_state, "astype") else next_state
                except Exception:
                    next_obs_serial = []

                # finalize the oldest pending (FIFO)
                with self._pending_lock:
                    if self._pending_actions:
                        p = self._pending_actions.pop(0)
                        transition = {
                            "obs": p["obs"],
                            "action": p["action"],
                            "reward": float(reward),
                            "next_obs": next_obs_serial,
                            "done": bool(done),
                            "logp": p.get("logp", None),
                            "value": p.get("value", None),
                            "mask": p.get("mask", None),
                            "meta": {"timestamp": p.get("timestamp", None)}
                        }
                        # push to buffer
                        self.replay_buffer.add(transition)
                        print(transition)
                        self.logger.info(f"[Actor] pushed transition (reward={reward}) to buffer (buffer_size={len(self.replay_buffer)})")
                        # optionally upload to learner over HTTP if configured (implement Learner endpoint)
                        if self.learner_http_url:
                            threading.Thread(target=self._http_upload_single, args=([transition],), daemon=True).start()

            elif operation == "GameResult":
                # finalize any remaining pending transitions as terminal with final reward if provided
                final_reward = float(msg_data.get("final_reward", 0.0)) if msg_data.get("final_reward") is not None else 0.0
                done = True
                # build next_obs as empty or from msg_data
                try:
                    next_state, _ = convert_message_to_state(msg_data, msg_data.get("cards", []), self.played_cards,
                                                             self.up_player_played, self.teammate_played,
                                                             self.others_played1, self.others_played2, self.others_played3,
                                                             self.remaining_counts_others, self.wild_cards)
                    next_obs_serial = next_state.astype(float).tolist() if hasattr(next_state, "astype") else next_state
                except Exception:
                    next_obs_serial = []

                with self._pending_lock:
                    while self._pending_actions:
                        p = self._pending_actions.pop(0)
                        transition = {
                            "obs": p["obs"],
                            "action": p["action"],
                            "reward": float(final_reward),
                            "next_obs": next_obs_serial,
                            "done": bool(done),
                            "logp": p.get("logp", None),
                            "value": p.get("value", None),
                            "mask": p.get("mask", None),
                            "meta": {"terminal": True, "timestamp": p.get("timestamp", None)}
                        }
                        self.replay_buffer.add(transition)
                        self.logger.info(f"[Actor] GameResult processed, pending cleared, buffer_size={len(self.replay_buffer)}")
                        # optionally notify remote learner as above
                        if self.learner_http_url:
                            threading.Thread(target=self._http_upload_single, args=(None,), daemon=True).start()

            # housekeeping: drop old pendings (timeout)
            self._drop_stale_pendings()

        except json.JSONDecodeError:
            self.logger.error("Actor: JSON decode error in handle_message")
        except Exception as e:
            self.logger.error(f"Actor handle_message exception: {e}")

    def _drop_stale_pendings(self):
        """Drop pending entries older than pending_timeout (with zero reward), to avoid memory leak."""
        now = time.time()
        removed = 0
        with self._pending_lock:
            new_list = []
            for p in self._pending_actions:
                if now - p.get("timestamp", 0.0) > self.pending_timeout:
                    removed += 1
                else:
                    new_list.append(p)
            self._pending_actions = new_list
        if removed:
            self.logger.warning(f"[Actor] dropped {removed} stale pending actions")

    def _http_upload_single(self, transitions: Optional[List[Dict]]):
        """
        Robust HTTP uploader with retries and detailed logging.
        Called in a background thread by actor when learner_http_url is configured.

        Replaces the simpler uploader to help debug connectivity / payload issues.
        """
        if not self.learner_http_url:
            self.logger.warning("[Actor._http_upload_single] learner_http_url not configured")
            return

        import requests
        payload = {"type": "trajectory_batch", "actor_id": self.key, "batch": transitions or []}
        headers = {"Content-Type": "application/json"}
        max_retries = 3
        backoff = 1.0

        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(
                    f"[Actor._http_upload_single] uploading {len(payload['batch'])} transitions to {self.learner_http_url} (attempt {attempt})")
                resp = requests.post(self.learner_http_url, json=payload, headers=headers, timeout=8.0)
                # log response status and body (helpful for debugging)
                try:
                    txt = resp.text
                except Exception:
                    txt = "<no-body>"
                self.logger.info(f"[Actor._http_upload_single] HTTP {resp.status_code} response: {txt[:200]}")
                if resp.status_code == 200:
                    # success
                    return
                else:
                    # not success — log and retry
                    self.logger.warning(
                        f"[Actor._http_upload_single] non-200 response: {resp.status_code}, attempt {attempt}")
            except requests.exceptions.RequestException as re:
                self.logger.warning(f"[Actor._http_upload_single] request exception on attempt {attempt}: {re}")
            except Exception as e:
                self.logger.exception(f"[Actor._http_upload_single] unexpected exception on attempt {attempt}: {e}")

            # exponential backoff
            time.sleep(backoff)
            backoff *= 2.0

        # after retries failed
        self.logger.error(
            "[Actor._http_upload_single] upload failed after retries; payload dropped (you may want to queue/retry later)")


# -------------------------
# Convenience runner
# -------------------------
def run_actor(key: str, buffer: ReplayBuffer, learner_http_url: Optional[str] = None, device: str = 'cpu'):
    """
    Instantiate agent and actor client, then run the asyncio loop.
    """
    agent = SimpleAgent(state_dim=436, max_actions=5000, device=device)
    client = RLActorClient(key=key, agent=agent, buffer=buffer, learner_http_url=learner_http_url)

    # run client loop
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(client.run())
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, required=False, default="a1", help="actor key (e.g. a1)")
    parser.add_argument("--mode", type=str, default="remote", choices=["local", "remote"], help="local: use in-process ReplayBuffer; remote: POST to learner")
    parser.add_argument("--learner_url", type=str, default="http://127.0.0.1:8000/upload", help="remote learner POST url (if mode==remote)")
    args = parser.parse_args()

    # For simple local testing, create one in-process ReplayBuffer and run actor.
    # In distributed setup, you'd create one ReplayBuffer in Learner and share or use remote mode.
    if args.mode == "local":
        buf = ReplayBuffer()
        run_actor(args.key, buf, learner_http_url=None)
    else:
        # remote: still create a buffer for temporary storage, but set remote upload URL
        buf = ReplayBuffer()
        run_actor(args.key, buf, learner_http_url=args.learner_url)
