# learner/server.py
"""
Learner HTTP receiver + training runner.

Run this file to:
  - start a FastAPI server that accepts POST /upload to receive actor trajectory batches
  - start a PPOLearner.train loop that consumes the in-memory ReplayBuffer filled by incoming uploads

Actor expected payload (JSON):
{
  "type": "trajectory_batch",
  "actor_id": "a1",
  "batch": [
    {
      "obs": [...],           # list of floats (state vector)
      "action": 3,
      "reward": 0.0,
      "next_obs": [...],
      "done": false,
      "logp": null,
      "value": null,
      "mask": [...],          # list of 0/1
      "meta": {...}
    },
    ...
  ]
}

This implementation is minimal but production-usable with small extensions (auth, TLS, persistent queue, rate-limiting).
"""

import threading
import uvicorn
import time
import os
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import asyncio
import numpy as np

# Ensure these imports match your project layout
from storage.replay_buffer import ReplayBuffer
from learner import PPOLearner

# Configurable params
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8000
SAMPLES_PER_UPDATE = 100
FETCH_INTERVAL = 1.0
TOTAL_UPDATES = 10000
SAVE_EVERY = 20
DEVICE = "cuda:0"
STATE_DIM = 436
ACTION_DIM = 1000

app = FastAPI(title="LearnerReceiver")
buffer = ReplayBuffer()

# ---------- Pydantic models for validation ----------
class TransitionModel(BaseModel):
    obs: Optional[list] = None
    action: Optional[int] = None
    reward: Optional[float] = 0.0
    next_obs: Optional[list] = None
    done: Optional[bool] = False
    logp: Optional[float] = None
    value: Optional[float] = None
    mask: Optional[list] = None
    meta: Optional[dict] = None

class UploadModel(BaseModel):
    type: str
    actor_id: Optional[str] = None
    batch: Optional[List[TransitionModel]] = None

# ---------- HTTP endpoint ----------
@app.post("/upload")
async def upload_trajectories(payload: UploadModel):
    """
    Receive trajectory batch from actor(s). Payload validated against UploadModel.
    """
    if payload.type != "trajectory_batch":
        raise HTTPException(status_code=400, detail="Unsupported payload type")

    if not payload.batch:
        # nothing to add
        return {"status": "ok", "added": 0}

    # convert pydantic models to plain dicts for ReplayBuffer
    transitions = []
    for t in payload.batch:
        # ensure keys exist and that obs/mask are lists (not nested np types)
        transition = {
            "obs": list(t.obs) if t.obs is not None else [],
            "action": int(t.action) if t.action is not None else 0,
            "reward": float(t.reward) if t.reward is not None else 0.0,
            #"next_obs": list(t.next_obs) if t.next_obs is not None else [],
            "done": bool(t.done) if t.done is not None else False,
            #"logp": float(t.logp) if t.logp is not None else None,
            #"value": float(t.value) if t.value is not None else None,
            #"mask": list(t.mask) if t.mask is not None else None,
            #"meta": t.meta or {}
        }
        transitions.append(transition)

    # bulk extend buffer
    try:
        buffer.extend(transitions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add to buffer: {e}")

    return {"status": "ok", "added": len(transitions)}

# health
@app.get("/health")
async def health():
    return {"status": "ok", "buffer_size": len(buffer)}

# ---------- Helper: run uvicorn server in a thread ----------
def start_http_server_in_thread(host: str = HTTP_HOST, port: int = HTTP_PORT):
    """Start uvicorn with the FastAPI app object in a background thread."""
    def run():
        # pass the app object directly so uvicorn does not need to import by module path
        uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t

# ---------- Training runner ----------
def start_learner_training(buffer: ReplayBuffer,
                           device: str = DEVICE,
                           state_dim: int = STATE_DIM,
                           action_dim: int = ACTION_DIM,
                           samples_per_update: int = SAMPLES_PER_UPDATE,
                           fetch_interval: float = FETCH_INTERVAL,
                           total_updates: int = TOTAL_UPDATES,
                           save_every: int = SAVE_EVERY):
    """
    Instantiate PPOLearner and run training loop (blocking). This runs in main thread.
    """
    learner = PPOLearner(state_dim=state_dim, action_dim=action_dim, buffer=buffer, device=device,
                         save_every_updates=save_every)
    print("[Learner] Starting training loop")
    learner.train(total_updates=total_updates, fetch_interval=fetch_interval,
                  samples_per_update=samples_per_update, save_every=save_every)
    print("[Learner] Training finished")

# ---------- Main entrypoint ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("learner_server")
    parser.add_argument("--host", type=str, default=HTTP_HOST)
    parser.add_argument("--port", type=int, default=HTTP_PORT)
    parser.add_argument("--state-dim", type=int, default=STATE_DIM)
    parser.add_argument("--action-dim", type=int, default=ACTION_DIM)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--samples-per-update", type=int, default=SAMPLES_PER_UPDATE)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--total-updates", type=int, default=TOTAL_UPDATES)
    args = parser.parse_args()

    # start HTTP server (background thread)
    start_http_server_in_thread(host=args.host, port=args.port)
    print(f"[Learner] HTTP server started at http://{args.host}:{args.port}")

    # start training (will block)
    start_learner_training(buffer=buffer,
                           device=args.device,
                           state_dim=args.state_dim,
                           action_dim=args.action_dim,
                           samples_per_update=args.samples_per_update,
                           total_updates=args.total_updates,
                           save_every=args.save_every)
