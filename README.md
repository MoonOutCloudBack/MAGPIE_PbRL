# Multi-Agent Reinforcement Learning via Agent-Specific Preference

This is the official implementation of MAGPIE (IEEE TASE 2026, [arxiv](https://arxiv.org/abs/2608.08604)). 

## Installation

```bash
conda create -n magpie python==3.8
conda activate magpie
# then, please install pytorch through https://pytorch.org/get-started/previous-versions/
# for cuda 12.2
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

pip install numpy pandas matplotlib wandb setproctitle
pip install gym==0.17.2 seaborn
pip install absl-py tensorboard tensorboardX

# Install this repository from its root directory.
pip install -e .
```

## Quick Start

We support several [MPE tasks](https://mpe2.farama.org/mpe2/), including `simple_spread`, `simple_speaker_listener`, and `simple_reference`, and a custom `mobile` task.

The following command starts MAGPIE on `simple_spread`:

```bash
python -m offpolicy.scripts.train.train_mpe \
  --scenario_name simple_spread --num_agents 3 --num_landmarks 3 \
  --algorithm_name qmix --experiment_name magpie \
  --use_PbRL --size_segment 5
```

Remove `--use_PbRL` to train the QMIX baseline with environment rewards. 

To run the custom mobile task:

```bash
python -m offpolicy.scripts.train.train_mobile \
  --algorithm_name qmix --experiment_name mobile_magpie \
  --use_PbRL --size_segment 5
```

## Acknowledgement

This repo benefits from [Off-Policy MARL Algorithms](https://github.com/marlbenchmark/off-policy) and [BPref](https://github.com/rll-research/BPref).
Thanks for their wonderful work.

## Citation

If you find this project helpful, please consider citing the following paper:

```bibtex
@article{mu2026magpie,
  title={{M}ulti-{A}gent {R}einforcement {L}earning via
{A}gent-{S}pecific {P}reference},
  author={Mu, Ni and Luan, Yao and Yang, Yiqin and Jia, Qing-Shan},
  journal={IEEE Transactions on Automation Science and Engineering},
  year={2026},
  publisher={IEEE}
}
```


