# 🧭 Maze Runner RL (PPO + GPU Training)

A reinforcement learning project that trains PPO agents to navigate procedurally generated mazes using partial observations. Features GPU-accelerated training, custom CNN policies, and comprehensive evaluation metrics.

## 🚀 Quick Start with uv (Recommended)

```bash
# Clone and navigate to project
cd maze_runner

# Activate the pre-configured virtual environment
source .venv/bin/activate

# Install/update dependencies with uv
uv pip install -r requirements.txt

# Install PyTorch with CUDA support (for GPU training)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Launch Jupyter Lab
jupyter lab maze_runner_rl.ipynb
```

## 🐍 Alternative Setup (pip/conda)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Launch notebook
jupyter lab maze_runner_rl.ipynb
```

## 🎯 Features

- **GPU-Accelerated Training**: Utilizes CUDA for fast CNN policy training
- **Procedural Maze Generation**: DFS backtracker algorithm creates unique mazes
- **Partial Observability**: 7×7 egocentric view with 4-channel observations
- **Smart Reward Shaping**: Distance-based rewards encourage goal-seeking behavior
- **A* Baseline Comparison**: Evaluates agent performance vs optimal paths
- **Real-time Monitoring**: Custom callbacks show training progress and success rates
- **Video Generation**: Creates GIFs of trained agent performance

## 🏃‍♂️ Training Tips

- **GPU Memory**: Uses ~3GB VRAM on RTX 2070, room for larger batch sizes
- **Timesteps**: Start with 500K timesteps, scale to 1M+ for better performance
- **Maze Difficulty**: Increase `size` to 21+ once achieving >70% success on 15×15
- **Memory**: Try `CnnLstmPolicy` for better performance in complex mazes
- **Monitoring**: Watch TensorBoard with `tensorboard --logdir tb_logs`

## 📁 Project Structure

```
maze_runner/
├── maze_runner_rl.ipynb    # Main training notebook (fixed runtime errors)
├── requirements.txt        # Python dependencies
├── .venv/                 # Pre-configured virtual environment
├── tb_logs/               # TensorBoard logs (created during training)
├── videos/                # Generated demo videos
└── README.md              # This file
```

## 🔧 Recent Improvements

- ✅ Fixed SubprocVecEnv multiprocessing errors
- ✅ Improved reward structure (10x goal reward, better shaping)
- ✅ Enhanced training monitoring with success rate tracking
- ✅ Optimized hyperparameters for GPU training
- ✅ Added smart evaluation with progress validation
- ✅ Reduced output noise (shows only last 5 training updates)

## 🎮 Usage

1. **Run all cells** in the notebook from top to bottom
2. **Monitor training** progress via the custom callback output
3. **Check results** in the evaluation section
4. **View agent behavior** in the generated GIF (`videos/ppo_maze_demo.gif`)
5. **Analyze metrics** in TensorBoard for detailed insights

The notebook is now fully functional and should train successfully without runtime errors!
