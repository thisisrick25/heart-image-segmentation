# Kaggle Workflow Setup Guide

## 🎯 Complete Workflow: Code Locally → GitHub → Train on Kaggle

This guide will help you set up a seamless workflow to:

1. Develop and test code locally on your machine
2. Version control your code with GitHub
3. Run heavy training jobs on Kaggle with free GPU/TPU

---

## 📋 Prerequisites

- Git installed on your machine
- Python 3.8+ installed
- GitHub account
- Kaggle account

---

## 🔧 One-Time Setup

### 1. Set Up Kaggle API

1. **Get your Kaggle API credentials:**

   - Go to https://www.kaggle.com/settings/account
   - Scroll to "API" section
   - Click "Create New Token"
   - This downloads `kaggle.json` file

2. **Place the credentials file:**

   - **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **Mac/Linux:** `~/.kaggle/kaggle.json`
   - **Set permissions (Mac/Linux only):**
     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```

3. **Install Kaggle CLI:**

   ```bash
   pip install kaggle
   ```

4. **Verify installation:**
   ```bash
   kaggle --version
   ```

### 2. Configure Your Kernel

Edit `kernel-metadata.json` in your project:

```json
{
  "id": "your-username/your-kernel-name",
  "title": "Heart Segmentation UNet Training",
  "code_file": "kaggle_train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": false,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

**Important:** Update the `id` field with your Kaggle username!

---

## 🔄 Daily Workflow

### Step 1: Code Locally

Work on your project files in your IDE:

```bash
# Edit your training script
code kaggle_train.py

# Test preprocessing
python preprocess.py

# Make changes to configuration
code config.py
```

### Step 2: Test Locally (Optional)

If you have a GPU locally, test your changes:

```bash
python train.py
```

### Step 3: Commit to GitHub

```bash
# Check what files changed
git status

# Add your changes
git add kaggle_train.py config.py requirements.txt

# Commit with a meaningful message
git commit -m "Updated training hyperparameters"

# Push to GitHub
git push origin main
```

**Pro Tip:** The `.gitignore` file ensures you don't accidentally commit:

- Large dataset files
- Model checkpoints
- Cache files

### Step 4: Push to Kaggle

**Option A: Use the helper script (Windows)**

```bash
setup_kaggle.bat
```

Then select option 1 to push.

**Option B: Manual push**

```bash
kaggle kernels push
```

This uploads:

- `kaggle_train.py` (your main script)
- `requirements.txt` (dependencies)
- `kernel-metadata.json` (configuration)

### Step 5: Run on Kaggle

1. **Open your kernel** in browser:

   ```
   https://www.kaggle.com/code/<your-username>/<your-kernel-name>
   ```

2. **Enable GPU:**

   - Click "Edit" button
   - On the right panel, under "Settings"
   - Select "Accelerator" → "GPU T4 x2" (or P100)
   - Click "Save"

3. **Run the kernel:**
   - Click "Save & Run All" button
   - Watch the logs stream in real-time
   - Training runs with free GPU! 🎉

### Step 6: Monitor & Download Results

**Monitor progress:**

```bash
# Check kernel status
kaggle kernels status your-username/your-kernel-name

# View output logs
kaggle kernels output your-username/your-kernel-name
```

**Download results:**

```bash
# Download all outputs (model, plots, logs)
kaggle kernels output your-username/your-kernel-name -p ./kaggle_output
```

Your trained model will be in: `./kaggle_output/best_model.pth`

---

## 🎓 Advanced Tips

### Working with Datasets on Kaggle

If you want to use a dataset that's already on Kaggle:

1. Find the dataset on Kaggle (e.g., "Task02 Heart Dataset")
2. Add it to your kernel-metadata.json:

```json
{
  ...
  "dataset_sources": ["username/dataset-name"],
  ...
}
```

3. In your `kaggle_train.py`, access it:

```python
DATASET_DIR = Path("/kaggle/input/task02-heart/Task02_Heart")
```

### Using Git Branches for Experiments

```bash
# Create a new branch for experiments
git checkout -b experiment-new-architecture

# Make changes and test
# ...

# Commit and push
git commit -am "Testing ResNet encoder"
git push origin experiment-new-architecture

# Push to Kaggle from this branch
kaggle kernels push
```

### Automated Training Runs

Create a script to automate the workflow:

```bash
# commit_and_train.bat
@echo off
git add .
git commit -m "%1"
git push origin main
kaggle kernels push
echo "✓ Code pushed to GitHub and Kaggle!"
```

Usage:

```bash
commit_and_train.bat "Updated learning rate"
```

---

## 🔍 Useful Kaggle Commands

```bash
# List all your kernels
kaggle kernels list --mine

# Check kernel status
kaggle kernels status username/kernel-name

# View kernel output/logs
kaggle kernels output username/kernel-name

# Download kernel outputs
kaggle kernels output username/kernel-name -p ./output

# Pull kernel code (download from Kaggle)
kaggle kernels pull username/kernel-name

# Cancel a running kernel
kaggle kernels status username/kernel-name
# (no direct cancel command via CLI, use web UI)
```

---

## 📊 File Organization

```
Your Local Repository
├── kaggle_train.py        ← Main training script (pushed to Kaggle)
├── train.py               ← Local training script
├── config.py              ← Configuration
├── requirements.txt       ← Dependencies (pushed to Kaggle)
├── kernel-metadata.json   ← Kaggle kernel config
├── setup_kaggle.bat       ← Helper script
├── .gitignore             ← Prevents committing large files
└── datasets/              ← NOT in Git (too large)

Your GitHub Repository
├── kaggle_train.py
├── train.py
├── config.py
├── requirements.txt
├── kernel-metadata.json
└── README.md

Your Kaggle Kernel
├── kaggle_train.py        ← Your script runs here
├── requirements.txt       ← Auto-installed
└── /kaggle/
    ├── input/             ← Read-only datasets
    └── working/           ← Your outputs go here
        ├── datasets/      ← Downloaded datasets
        └── result/        ← Model checkpoints
```

---

## 🐛 Troubleshooting

### "401 Unauthorized" error

- Check if `kaggle.json` is in the correct location
- Regenerate API token from Kaggle settings

### "Kernel not found" error

- Update the `id` in `kernel-metadata.json`
- Use format: `your-username/kernel-slug`

### Out of memory on Kaggle

- Reduce `BATCH_SIZE` in your script
- Use smaller `SPATIAL_SIZE`
- Check GPU usage with: `nvidia-smi` in a Kaggle notebook

### Dataset not downloading on Kaggle

- Enable "Internet" in kernel settings
- Add dataset to `dataset_sources` in metadata
- Use Kaggle's built-in datasets when possible

---

## 🎉 Quick Reference

**Push code to Kaggle:**

```bash
kaggle kernels push
```

**Check training status:**

```bash
kaggle kernels status your-username/your-kernel
```

**Download trained model:**

```bash
kaggle kernels output your-username/your-kernel -p ./results
```

**Commit and push to GitHub:**

```bash
git add .
git commit -m "Your message"
git push origin main
```

---

## 📚 Additional Resources

- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [Kaggle Kernels Guide](https://www.kaggle.com/docs/kernels)
- [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics)
- [GitHub Flow](https://guides.github.com/introduction/flow/)

---

Happy coding! 🚀
