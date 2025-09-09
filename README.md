# STM Tracking Setup

This repository contains the STM tracking project with Python-based analysis tools.

## Prerequisites

For a completely new computer, you'll need to install:

1. **Python Distribution**: Download Anaconda distribution with conda and pip
   - [Download Anaconda](https://www.anaconda.com/download/success)
   - *(Note: We prefer conda virtual environments, but other package managers work too)*

2. **Git**: Download and install Git
   - [Download Git](https://git-scm.com/downloads)

## Setup Instructions

### 1. Create Conda Environment

Open your terminal and run the following commands:

```bash
conda init
conda create -n elm_env python=3.10
```

> **Note**: Python 3.10 is recommended. Higher versions might work but may require dependency adjustments.

You can replace `elm_env` with any environment name you prefer. Confirm with `y` when prompted.

### 2. Activate Environment

```bash
conda activate elm_env
```

> **Note**: You might need to close and reopen the terminal the first time, or run `conda init` again if activation fails.

### 3. Clone Repository

Navigate to your desired folder and clone the repository:

```bash
git init
git clone https://github.com/LC-Nguyen/STM_tracking_public.git
cd STM_tracking_public
```

### 4. Install Python Libraries

```bash
pip install --no-deps -r requirements.txt
```

### 5. Install TensorFlow

#### Windows and Intel Mac:
```bash
pip install tensorflow
```

#### Apple Silicon Mac:
```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

### 6. Setup Jupyter Kernel

```bash
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=elmloc310_kernel
```

This allows you to run the environment kernel in Jupyter notebooks.

## Known Issues

### Model Compatibility Issue

**Problem**: Model files (.h5) don't behave consistently between different machines (e.g., Mac vs. remote server).

**Temporary Solution**: Only use .h5 files that were trained on the same computer where they'll be executed.

**Permanent Solution**: Consider setting up a Docker container for cross-platform compatibility (not currently implemented due to project scale).

### NumPy Version Compatibility

**Problem**: As of June 2025, NumPy transitioned to version 2.0, which TensorFlow doesn't yet support.

**Solution**: Use the library versions specified in `requirements.txt` and be cautious when updating dependencies.

### TensorFlow on Mac Installation

**Problem**: TensorFlow installation may fail on Mac due to ARM architecture issues with incorrect Python versions.

**Diagnosis**: Check your architecture compatibility:

```bash
# Check system architecture
uname -m
# Should return: arm64 (Apple Silicon) or x86_64 (Intel)

# Check Python architecture
file $(which python)
# Should return: Mach-O 64-bit executable arm64 (Silicon) or x86_64 (Intel)
```

**Solution**: If the results don't match, you have a version mismatch. Uninstall and reinstall Python with the correct architecture.

### Runtime Warnings

#### m.analyze Warning

**Problem**:
```
RuntimeWarning: invalid value encountered in scalar divide
```

**Cause**: Running analysis with only one dataset per MotionAnalyzer object.

**Impact**: Warning only, doesn't affect functionality.

#### dp.plot_scatter_rot Warning

**Problem**:
```
UserWarning: The palette list has more values (3) than needed (2)
```

**Cause**: Missing certain rotation types (translated, rotated, or no movement).

**Impact**: Warning only, doesn't affect functionality.

### frame_correct.py Limitations

**Problem**: Doesn't work well without consistent and clear backgrounds (like moiré patterns).

**Solution**: Ensure clear, consistent backgrounds for proper alignment. Rule of thumb: if you can't visually determine background alignment, `frame_correct.py` likely can't either.

## Getting Started

After setup, activate your environment and start working:

```bash
conda activate elm_env
jupyter notebook
```
You can use jupyter lab instead if you prefer.

Select the `elmloc310_kernel` when creating or opening notebooks.

## Support

If you encounter issues not covered in this README, please check the repository's issue tracker or create a new issue with detailed information about your setup and the problem encountered.
