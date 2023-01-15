# RNNPool

Repository for project 03 in DLP (2022 Fall) in USTC.

Modification and experiments on RNNPool.

[Paper Link](https://arxiv.org/abs/2002.11921)	[Original Repository Link](https://github.com/microsoft/EdgeML)

## Folder Structure

```
.
├── README.md
├── docs
│   ├── dlp2022秋第三次作业要求.pdf
│   └── paper
│       ├── imgs
│       ├── rnnpool_NeurIPS2020.pdf   # rnnpool paper
│       ├── rnnpool_arxiv.pdf         # rnnpool paper
│       ├── translation.md            # translation for abstract and introduction
│       └── translation.pdf           # translation for abstract and introduction
├── result
│   └── vww
│       ├── graph                     # graph for results, such as loss curve
│       ├── modified                  # log info for modified models
│       └── original                  # log info for original model
└── src
    ├── Visual_Wakeword                        # code for visual wakeword task
    │   ├── modified                           # train and eval using modified models
    │   │   ├── checkpoints                    # checkpoints for modified models
    │   │   ├── eval_cpu_modified.py           # code for eval
    │   │   ├── modified_models                # store modified models
    │   │   └── train_vww_modified.py          # train modified models
    │   └── original                           # train and eval using original model
    │       ├── README.md
    │       ├── checkpoints                    # checkpoints for original models
    │       ├── eval_cpu.py                    # code for eval
    │       ├── model_mobilenet_rnnpool.py     # original model
    │       ├── requirements.txt
    │       ├── scripts                        # scripts for downloading dataset
    │       └── train_visualwakewords.py       # train original model
    └── env_setup                              # code for environment setup
        ├── README.md
        ├── bitahub_envsetup_include_vww.sh    # shell script for bitahub environment setup
        ├── edgeml_pytorch                     # python package
        ├── requirements-bitahub-include-vww.txt
        ├── requirements-cpu.txt
        └── setup.py
```
