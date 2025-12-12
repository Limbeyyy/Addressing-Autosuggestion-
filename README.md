# Hybrid TinyCNN + TFLite Autocomplete Project

This repo contains a pipeline to train a tiny char-CNN classifier for prefix/autocomplete ranking and convert it to a quantized TFLite model for mobile use. It also includes Trie-based prefix filtering and a Python runtime to test suggestions.

## Quick start (Python)

1. Create a virtual env and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r python/requirements.txt
   ```

2. Prepare data (place your words in `data/words.csv` and optional real typos in `data/real_typos.csv`):
   ```bash
   python python/prepare_data.py --aug_per_seed 20 --test_count 50
   ```

3. Train the tiny CNN:
   ```bash
   python python/train_cnn.py --max_len 12 --epochs 12 --batch 128
   ```

4. Convert to TFLite (quantized int8):
   ```bash
   python python/convert_tflite.py
   ```

5. Build trie & labels:
   ```bash
   python python/build_trie.py
   ```

6. Run the Python autocomplete server for interactive testing:
   ```bash
   python python/autocomplete_server.py
   ```

## Android integration (Java)
- Copy `models/model.tflite`, `python/labels.txt`, and `python/artifacts/trie.json` into `android/app/src/main/assets/`.
- Use the provided `TFLiteRanker` Java helper to perform inference and rank candidates returned by the Trie.

## Notes
- Replace augmentation logic with Nepali keyboard neighbor map for better results.
- TFLite conversion uses representative dataset (`data/repr_samples.txt`) for full-int8 quantization.
- The model and artifacts are intentionally small to meet the 50–150 KB target after quantization.
