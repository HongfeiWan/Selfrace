<p align="center">
    <a href="https://github.com/HongfeiWan/Selfrace" target="_blank">
        <img src="https://github.com/HongfeiWan/Selfrace/blob/main/images/Logo.png" width="1000">
    </a>
</p>
<p align="center">
    <a href="https://github.com/pytorch/pytorch">
        <img src="https://img.shields.io/badge/pytorch-2.8.0-brightgreen.svg">
    </a>
    <a href="https://github.com/NVIDIA/cuda-python">
        <img src="https://img.shields.io/badge/cudapython-13.0.1-brightgreen.svg">
    </a>
</p>

Selfrace is a simulator for self-play end-to-end auto-driving training.

## Installation

```bash
# Clone the repository
git clone https://github.com/HongfeiWan/Selfrace.git
cd Selfrace

# Install dependencies
pip install -r requirements.txt
```

## Documentation

For detailed documentation, tutorials, and API references, please visit our [Wiki](https://github.com/HongfeiWan/Selfrace/wiki).

The wiki contains comprehensive information about:
- Core modules and their usage
- Configuration guides
- API references
- Performance optimization
- Troubleshooting guides
- And much more!

## Quick Start

1. Follow the installation instructions above
```bash
# Training
cd Selfrace/training
python ddppo.py
```

## License

This project is licensed under the GNU General Public License v3.0 License - see the [LICENSE](LICENSE) file for details.

## References
- [Self-Play Reinforcement Learning for Autonomous Driving](https://arxiv.org/abs/2502.03349)
