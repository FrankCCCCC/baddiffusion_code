# BadDiffusion

## Environment

- Python 3.8.5
- PyTorch 1.10.1+cu11 or 1.11.0+cu102

## Usage

### Install Require Packages and Prepare Essential Data

Please run

```bash
bash install.sh
```

### Wandb Logging Support

If you want to upload the experimental results to ``Weight And Bias``, please login with API key.

```bash
wandb login --relogin --cloud <API Key>
```

### Prepare Dataset

- CIFAR10: It will be downloaded by HuggingFace ``datasets`` automatically
- CelebA-HQ: Dowload the CelebA-HQ dataset and put the images under the folder ``./datasets/celeba_hq_256``

### Run BadDiffusion

### Run Adversarial Neuron Pruning (ANP)


