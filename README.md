# Protein Suppressor Predictor

A machine learning-based predictor for identifying protein suppressors.

## Project Structure

```
protein_suppressor_predictor/
├── README.md                         # Project overview + Windsurf setup
├── requirements.txt                  # Python packages needed
├── .windsurfignore                   # Files for Windsurf to ignore
├── .gitignore                        # Git ignore (includes data/)
│
├── data/                            # All your data (gitignored)
│   ├── proteins/                    # Raw protein sequences
│   ├── structures/                  # PDB/AlphaFold structures  
│   ├── features/                    # Processed features
│   └── results/                     # Model outputs
│
├── src/                            
│   ├── data_collection.py           # Download from UniProt, PDB, etc.
│   ├── feature_extraction.py        # Calculate all features
│   ├── model_training.py            # Train your ML models
│   ├── prediction.py                # Make predictions
│   ├── visualization.py             # Create plots and figures
│   └── utils.py                     # Helper functions
│
├── notebooks/                       # Jupyter notebooks for analysis
│   ├── 01_explore_data.ipynb
│   ├── 02_build_features.ipynb  
│   ├── 03_train_models.ipynb
│   └── 04_results_analysis.ipynb
│
├── models/                          # Saved trained models
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   └── best_model.pkl
│
├── figures/                         # All your plots and visualizations
│   ├── feature_importance.png
│   ├── roc_curves.png
│   ├── protein_structures/
│   └── results_summary.png
│
├── config.py                        # All settings in one place
└── main.py                          # Run everything from here
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Initialize the project:
```bash
python main.py
```

## Data Collection

The data directory structure is designed to handle raw protein sequences, structural data, and processed features. All data should be placed in the appropriate subdirectories within the `data/` folder.

## Development

This project uses Jupyter notebooks for exploratory data analysis and model development. The main Python scripts in the `src/` directory handle the core functionality.
