## Brief overview
This rule file provides guidelines for developing and maintaining the gomoku reinforcement learning project. The project uses Python with PyTorch and follows a modular architecture for training AI agents to play gomoku (五子棋).

## Project architecture
- Follow the existing modular structure with separate directories for game logic, actors, and RL algorithms
- Keep game environment (gomoku_env.py) separate from training logic
- Use actor pattern for different player implementations (human, AI, random)
- Maintain clear separation between policy networks and training frameworks

## Coding conventions
- Use numpy arrays for board state representation (15x15 grid)
- Follow gym environment interface standards for reset(), step(), and render()
- Use PyTorch for neural network implementations
- Implement proper logging with timestamps and levels
- Use type hints where appropriate for better code clarity

## Training workflow
- Support both "init" (from scratch) and "continue" (from checkpoint) training modes
- Implement evaluation intervals during training to monitor progress
- Save model checkpoints at regular intervals
- Use command-line arguments for flexible training configuration

## Game logic standards
- Board representation: 0=empty, 1=black, 2=white
- Implement proper win condition checking in all directions (horizontal, vertical, diagonal)
- Handle edge cases like full board (draw) and invalid moves
- Maintain current player state and game over flags

## Visualization and debugging
- Use pygame for human-readable game visualization
- Implement probability visualization for AI move predictions
- Include proper error handling and validation
- Provide clear status messages during training and gameplay

## File organization
- Keep main training scripts at project root level
- Organize related modules in subdirectories (gomoku/, actors/, rl/)
- Use __init__.py files for proper Python package structure
- Store model checkpoints in dedicated model/ directory
