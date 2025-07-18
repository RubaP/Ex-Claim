
# Ex‑Claim 🚩

**Entity-aware Cross‑lingual Claim Detection for Automated Fact‑Checking**

---

## 🔍 Overview

**Ex‑Claim** introduces an entity-aware model for cross-lingual verifiable claim detection. Leveraging named entity recognition and entity linking, this model effectively identifies factual claims in multilingual social media posts—even in languages unseen during training. Its design supports robust knowledge transfer across languages.  
[📄 View the paper on arXiv](https://arxiv.org/pdf/2503.15220)

---

## ✨ Key Features

- **Entity-aware prediction**: Integrates multilingual Named Entity Recognition (MultiNERD) and Entity Linking (mGENERE) to include entity type and popularity signals.
- **Cross-lingual generalisation**: Fine-tuned with XLM‑R embeddings, achieving consistent gain across 27 languages, including unseen ones during training.
- **Modular architecture**: Offers variants—**X‑Claim** (baseline), **EXN‑Claim** (add entity types), and **EXP‑Claim** (add entity popularity).

---

## 📚 Research Insight

A parallel paper details the model and experiments:

- **Title**: *Entity-aware Cross-lingual Claim Detection for Automated Fact-checking*  
- **Authors**: R. Panchendrarajan & A. Zubiaga  
- **Published**: arXiv, 2025  
- **Link**: [arxiv.org/abs/2503.15220](https://arxiv.org/abs/2503.15220)

---

## 🚀 Installation

```bash
git clone https://github.com/RubaP/Ex-Claim.git
cd Ex-Claim
pip install -r requirements.txt
```

---

## 🧠 Model Variants

| Variant      | Components                               | Description                                       |
|--------------|------------------------------------------|---------------------------------------------------|
| **X‑Claim**   | XLM‑R embeddings                          | Multilingual baseline without entity features     |
| **EXN‑Claim** | XLM‑R + Named-entity type embeddings      | Adds entity-type awareness                        |
| **EXP‑Claim** | XLM‑R + Entity type & popularity embeddings | Adds entity popularity from Entity Linking                  |

---

## 📊 Evaluation & Results

- Tested across **3 datasets** covering **27 languages**, including synthetic data via machine translation.
- **EXP‑Claim** consistently outperformed baselines like mBERT, XLM‑R, and mT5.
- Achieved **~87% cross-lingual transferability**, outperforming the baselines.

---

## 🧪 Datasets

- **CheckThat! 2022** (Twitter) - COVID-19 tweets written in 5 languages
- **Synthetic Data** – synthetically translation of CheckThat! Test data across 18 languages
- **Kazemi** (WhatsApp Tiplines) – COVID-19 and Political posts written in 5 languages

---

## 🚧 Known Limitations

- **COVID-19 domain bias** – most training data are COVID-related tweets.
- **NER/EL limitations** – model performance can degrade with poor tagging in low-resource languages.
- **False predictions** – struggles with personal-experience phrasing or visual/multimodal claims.

---


## 🗂️ Project Structure

The repository is organized modularly under a `src/` directory, with distinct folders handling different components of the claim detection pipeline:

```
src/
├── analysis/         # Scripts for analyzing model behavior and errors
├── data/             # Scripts for synthetic data generation
├── evaluation/       # Claim detection logic and entity linking tuning
├── models/           # Core model architecture for claim detection
├── util/             # Utility modules: NER, entity linking, training, evaluation, etc.
```

### 📁 analysis/
Scripts to analyze the behavior and performance of trained models:
- `analyse_attention_weights.py` – Visualize self-attention weights.
- `compute_attention_entropy.py` – Measure uncertainty via entropy.
- `error_analysis.py` – Examine false positives/negatives.
- `knowledge_transfer.py` – Study multilingual knowledge transfer.
- `visualize_entity_embeddings.py` – Plot entity embeddings in 2D.

### 📁 data/
Handles data generation:
- `generateSyntheticTestData.py` – Translates test data to new languages.

### 📁 evaluation/
Core scripts for model experimentation:
- `claim-detection.py` – Baselines and X-Claim for claim classification.
- `claim-detectionE.py` – EX-Claim Pipeline implementation.
- `entity-linking-parameter-tuning.py` – Optimizes EL thresholds.

### 📁 models/
Contains model architecture:
- `ClaimDetection.py` – Defines the neural network, attention mechanisms, and entity fusion layers.

### 📁 util/
Reusable components used across the project:
- `Embedding.py` – Handles word/entity embeddings.
- `EntityLinking.py` – Entity disambiguation and popularity lookup.
- `Evaluation.py` – Model performance metrics.
- `NER.py` – Multilingual named entity recognition interface.
- `Preprocess.py` – Tokenization, entity tagging, and input prep.
- `ReadDataset.py` – Dataset loaders and formatters.
- `Results.py` – Aggregates and saves experiment results.
- `Training.py` – Model training logic.
- `Util.py` – Miscellaneous helper functions.


## 📄 Citation

If you use **Ex‑Claim**, please cite:

```bibtex
@inproceedings{panchendrarajan2025exclaim,
  title={Entity-aware Cross-lingual Claim Detection for Automated Fact-checking},
  author={Panchendrarajan, Rruba and Zubiaga, Arkaitz},
  booktitle={arXiv preprint arXiv:2503.15220},
  year={2025}
}
```

---

## 📧 Contact

For questions or suggestions, open an issue or contact:

**Rruba Panchendrarajan** – r.panchendrarajan@qmul.ac.uk  
GitHub: [@RubaP](https://github.com/RubaP)

---
