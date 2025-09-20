# Maze Runner RL (PPO on GPU)

Quick start:
1. Create env and install PyTorch with CUDA from https://pytorch.org/get-started/locally/
2. `pip install -r requirements.txt`
3. Launch: `jupyter lab` and open `maze_runner_rl.ipynb`
4. Run cells top-to-bottom.

Training tips:
- Start with 1e6 timesteps, then scale to 5e6+ for stronger generalization.
- Increase `size` to 21 for harder mazes once 15x15 >80% success.
- Use `CnnLstmPolicy` for better memory under partial observability.
- Watch TensorBoard (`tensorboard --logdir tb_logs`).

Files:
- `maze_runner_rl.ipynb`: end-to-end training + eval + videos
- `requirements.txt`: Python deps (install torch separately with CUDA)
