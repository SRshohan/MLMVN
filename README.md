# MLMVN Neural Network Training Framework

## Overview

This repository contains an implementation of Multi-Level Multi-Valued Neuron (MLMVN) neural networks with Batch Least-Squares learning algorithms. The framework is designed for classifying complex-valued Fourier data and includes both single-batch and multi-batch learning approaches.

## Files

* `train_MLMVNc.m`: Main script for training using batched learning approach
* `Net_LearnC.m`: Single-batch learning for MLMVN networks
* `Net_learn_testcodeC.m`: Continued learning with pre-trained weights
* `Net_LearnL.m`: Alternative implementation requiring labels attached to features
* `Net_testC.m`: Testing function for evaluating trained models

## Setup

1. Place all `.m` files in the same directory.
2. Prepare your data:

   * Feature matrices should be in `.mat` format
   * For Fourier data, features should be phase data in radians \[0, 2π)
   * Data should be ordered by class (e.g., all class 0 samples first, then class 1, etc.)

## Usage

### Quick Start

Run the following command in MATLAB:

```matlab
train_MLMVNc
```

### Data Preparation

Ensure the input feature matrices are correctly formatted and saved in `.mat` files as specified in the Setup section.

### Configuration

Edit `train_MLMVNc.m` to set training parameters such as number of neurons, thresholds, and feature count.

## Training Process

The training process consists of:

1. **Initial Training**: The first 300 samples (class 0) are used to initialize the network.
2. **Batch Learning**: Remaining classes are learned in batches of 300 samples.
3. **Testing**: After each global iteration, the network is evaluated.
4. **Convergence**: Training stops when the recognition rate exceeds the predefined threshold.

## Detailed Parameters

| Parameter       | Description                                            | Typical Value |
| --------------- | ------------------------------------------------------ | ------------- |
| `hidneur_num`   | Number of hidden neurons                               | 512-1024      |
| `outneur_num`   | Number of output neurons (usually = number of classes) | 3-10          |
| `sec_nums`      | Number of sectors for each output neuron               | \[2,2,2]      |
| `RMSE_thresh`   | Global angular RMSE threshold                          | 1.04          |
| `local_thresh`  | Local angular error threshold                          | 0-0.1         |
| `feature_count` | Number of features to use                              | 1000          |

## Troubleshooting

* **Memory Issues**: Reduce `feature_count` or `hidneur_num`
* **Slow Convergence**: Increase `local_thresh` slightly (0.05-0.1)
* **Poor Recognition**: Try increasing `hidneur_num` or adjusting `sec_nums`
* **Dimension Errors**: Ensure input data has consistent dimensions

## Example Output

The output typically includes the recognition rate and RMSE after each batch, along with convergence status.

## Citation

If you use this code in your research, please cite:

```bash
E. Aizenberg, I. Aizenberg, "Batch LLS-based Learning Algorithm for MLMVN 
with Soft Margins", Proceedings of the 2014 IEEE Symposium Series of 
Computational Intelligence (SSCI-2014), December, 2014, pp. 48-55.
```

