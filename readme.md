
# 🚀 Title
This is a training example based on baichuan2. In addition to the official training code, it includes parallel inference code, which is rare to find in existing examples.

## 📋 Training Steps
Follow these steps to set up and run your training environment:

### 0. Install Dependencies
Before starting, make sure you have all the required libraries:
```bash
pip install -r requirements.txt
```

### 1. Dataset Splitting
First, split your dataset by navigating to the `data` folder and running:
```bash
python split.py
```

### 2. Test Dataset Processing
As this is a generic multi-turn dialogue, you'll need to break it into several single-turn dialogues for testing. Enter the `data_preprocess` folder and run:
```bash
sh preprocess.sh
```

### 3. Training
Configure the model's load and save paths in `fine-tune.sh`, then execute the script:
```bash
sh fine-tune.sh
```

### 4. Testing
Set the saved model's path in `inference.sh`, then run the script to test:
```bash
sh inference.sh
```

### 5. Viewing/Evaluating Results
The test results are saved in `output.json`. You can also modify the save path in `inference.sh`.

## 💡 Tricks & Tips
- **Hardware Requirements:** It's recommended to use 4 or more A100 (80G) GPUs for optimal performance.
- **Data Preprocessing:** Try to complete all data processing before model training. Avoid adding business logic into the training code unless necessary.
- **Padding in Tokenizer:** The tokenizer uses right-padding during training. Despite warnings suggesting that decoder-only architectures require left-padding, the prediction results are correct. This bug will be considered in future updates, and solutions are welcome!

---

We welcome contributions and solutions to improve this project. If you encounter any issues or have suggestions, please feel free to open an issue or submit a pull request. Happy training! 🎉
