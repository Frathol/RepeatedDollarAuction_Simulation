"""
smoke_mesa.py -- checks the Mesa wrapper end to end (needs mesa>=3.0, pandas).
    python -m experiments.smoke_mesa
"""
from src.simulation.config import SimConfig
from src.simulation.runner import run_with_mesa

cfg = SimConfig(T=200, seed=0)
model, model_df, agent_df = run_with_mesa(cfg, "exp3", use_ph=True, progress=True)
print(model_df.head())
print(agent_df.head())
print("rows:", len(model_df), len(agent_df))
assert len(model_df) == cfg.T and len(agent_df) == 2 * cfg.T
print("OK")