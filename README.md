# IHML  
## Incremental Heuristic Meta-Learner for Classification

IHML (Incremental Heuristic Meta-Learner) is a classification-oriented meta-learning framework that incrementally constructs an effective meta-learner by combining heuristic feature selection with multiple base learning algorithms.  
The framework is designed to reduce model complexity while preserving robustness and predictive performance across incremental training cycles.

This repository provides the **reference implementation** of the IHML algorithm as introduced in the *Applied Artificial Intelligence* journal.

---

## Key Features

- Incremental meta-learning architecture  
- Heuristic-driven automatic feature selection  
- Dynamic base-learner participation  
- Fully automated training and testing cycles  
- Designed for large-scale and evolving datasets  

---

## Algorithm Overview

IHML operates in an incremental fashion by:
1. Selecting informative feature subsets using heuristic rules  
2. Training multiple base classifiers on selected features  
3. Combining base learners through a meta-learner  
4. Updating the meta-learner iteratively as new data becomes available  

The process eliminates unnecessary features and weak learners without requiring manual intervention, improving both efficiency and generalization.

---

## Repository Structure

```
├── data/                     # Partial sample datasets (for structure only)
├── globals.py                # Global configuration file
├── incremental_metalearner   # Main execution script
├── base_learners/            # Base learning algorithms
├── utils/                    # Helper functions and utilities
└── README.md
```

**Important:**  
The `data/` directory contains **partial samples only**.  
Full datasets must be obtained from their original sources (e.g., UCLA, UCI, Kaggle).

---

## Setup & Configuration

### 1. Configure Global Settings

Edit `globals.py` and ensure that:
- All directory paths are correctly set  
- Dataset locations are valid  
- Output folders exist or can be created  

This step is mandatory before execution.

---

### 2. Enable Base Learners

In `globals.py`, explicitly enable the base algorithms to be included in the meta-learner, such as:
- Random Forest (RF)  
- XGBoost (XGB)  

Only the enabled learners will participate in the IHML process.

---

### 3. Advanced Configuration (Optional)

For advanced tuning of:
- Heuristic thresholds  
- Incremental update behavior  
- Feature selection parameters  

Please refer to the reference paper for detailed explanations.

---

## Running IHML

After configuration, start the incremental learning process by running:

```bash
incremental_metalearner
```

This command will automatically:
- Initialize the IHML pipeline  
- Start incremental training cycles  
- Perform testing and evaluation at each stage  

No additional manual steps are required.

---

## Datasets

This implementation has been evaluated on multiple benchmark datasets.  
Due to licensing and size constraints, datasets are **not distributed in full** within this repository.

Please download datasets directly from their original sources (e.g., UCLA, UCI, Kaggle) and place them under the `data/` directory.

---

## Reference

If you use this code in academic or industrial work, please cite:

Incremental Heuristic Meta-Learning for Classification  
Applied Artificial Intelligence, 2024  

https://www.tandfonline.com/doi/full/10.1080/08839514.2024.2434309

---

## License

This project is provided for **research and educational purposes**.  
Please consult the repository license file for detailed usage terms.

---

## Contributions

Contributions, bug reports, and extensions are welcome.  
For major changes, please open an issue first to discuss your proposal.
