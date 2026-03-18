# NLP Content Moderation with Transformers

## Problem Description

Content moderation is a critical challenge for modern social media platforms. Harmful posts — including hate speech and offensive language, can spread quickly, damage communities, and expose platforms to regulatory risk. Manual moderation is expensive, slow, and psychologically taxing for human reviewers.

This project builds an automated content moderation prototype for **SafeSpace AI**, a hypothetical startup developing moderation tools for mid-size social networks and community forums. The system classifies social media posts into three categories:

- **Hate Speech** — content that attacks individuals or groups based on protected characteristics
- **Offensive Language** — content that is rude, vulgar, or inappropriate but does not target a protected group
- **Neither** — content that is acceptable and does not require moderation action

The critical challenge is distinguishing hate speech from merely offensive language. This distinction has legal implications — platforms that fail to remove hate speech face regulatory penalties, while over-censoring offensive (but legal) speech damages user trust.

The project compares a traditional ML baseline (**TF-IDF + Logistic Regression**) against a fine-tuned transformer model (**DistilBERT**) to evaluate whether the added complexity of deep learning is justified for this task. It also includes error analysis, confidence-based thresholds, and a production moderation workflow design.

## Dataset Source

The project uses the **Twitter Hate Speech and Offensive Language Dataset** created by Davidson et al. (2017).

- **Source**: [Davidson et al. on GitHub](https://github.com/t-davidson/hate-speech-and-offensive-language)
- **Direct CSV**: `labeled_data.csv`

**Dataset Statistics:**

| Property | Value |
|----------|-------|
| **Total tweets** | 24,783 |
| **Classes** | 3 (hate speech, offensive, neither) |
| **Class 0 — Hate Speech** | 1,430 (5.8%) |
| **Class 1 — Offensive Language** | 19,190 (77.4%) |
| **Class 2 — Neither** | 4,163 (16.8%) |
| **Annotation source** | CrowdFlower workers |

The dataset is **heavily imbalanced** (~77% offensive), which makes hate speech detection particularly challenging. Text includes slang, abbreviations, URLs, @mentions, hashtags, and emoji.

> **Note:** This dataset contains offensive and hateful language. This is necessary for building moderation systems. Approach the content professionally.

## Setup and Run Instructions

### Prerequisites

- Python 3.8+
- Google Colab (recommended) or a local machine with GPU support

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sepehrman/mini-project-9.git
   cd mini-project-9
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Key dependencies include:
   - `transformers` — Hugging Face DistilBERT
   - `torch` — PyTorch training loop
   - `scikit-learn` — TF-IDF baseline, metrics, GridSearchCV
   - `datasets` — Data handling
   - `accelerate` — Training utilities
   - `matplotlib` / `seaborn` — Visualization
   - `wordcloud` — Text exploration

3. **Run the notebook:**

   Open `notebooks/Mini_Project_9.ipynb` in Google Colab or Jupyter Notebook and run all cells. The dataset is downloaded automatically from GitHub within the notebook.

### Project Structure

```
mini-project-9/
├── notebooks/
│   └── notebook.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Results Summary

### Model Comparison

| Metric | TF-IDF + LR | DistilBERT |
|--------|-------------|------------|
| **Accuracy** | 0.8650 | 0.8687 |
| **Macro F1** | 0.7223 | 0.7341 |
| **Weighted F1** | 0.8738 | 0.8792 |

### Per-Class F1 Scores

| Class | TF-IDF + LR | DistilBERT | Δ |
|-------|-------------|------------|---|
| **Hate Speech** | 0.4195 | 0.4211 | +0.0016 |
| **Offensive** | 0.9170 | 0.9163 | −0.0006 |
| **Neither** | 0.8306 | 0.8649 | +0.0343 |

### Key Findings

1. **DistilBERT outperformed the baseline** across all aggregate metrics, with the largest improvement on the Neither class (+3.4% F1).
2. **Hate speech remained the hardest class** for both models (~0.42 F1), due to severe class imbalance (5.8% of data) and semantic overlap with offensive language.
3. **A model predicting "offensive" for everything would score 77% accuracy** — per-class F1 is the meaningful metric, not accuracy.
4. **Error analysis** revealed that most failures involved Hate/Offensive confusion (keyword-triggered misclassification) and context-dependent errors (sarcasm, slang, quotation).
5. **Production workflow**: With class-specific confidence thresholds (0.90 for auto-removal, 0.85 for flagging/approval), approximately 95% of posts can be handled automatically, with ~5% routed to human moderators.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | `distilbert-base-uncased` |
| Epochs | 5 |
| Batch size | 32 |
| Learning rate | 2e-5 |
| Optimizer | AdamW (weight decay 0.01) |
| Max sequence length | 128 tokens |
| Loss | Weighted CrossEntropyLoss (inverse-frequency) |
| Sampling | WeightedRandomSampler |

## Team Member Contributions

| Member | Contributions |
|--------|--------------|
| **Sepehr Mansouri** | Data exploration and preprocessing, DistilBERT fine-tuning and training loop, confidence analysis and production workflow design, report refining, F-IDF + Logistic Regression baseline with GridSearchCV, model comparison |
| **Tanishq Rawat** | Report writing, readme & Per-class analysis, error analysis and failure categorization |

## References

- Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). *Automated hate speech detection and the problem of offensive language.* Proceedings of ICWSM. [Paper](https://ojs.aaai.org/index.php/ICWSM/article/view/14955)
- [Twitter Hate Speech and Offensive Language Dataset — Direct CSV](https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv)
