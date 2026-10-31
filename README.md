# IsoVita: Manuscript Analyses

**Citation:** [TBD]

This repository contains the analysis scripts and Jupyter notebooks used to generate the results and figures for the IsoVita manuscript. 

## Software package

The IsoVita Python package is maintained separately:

- **GitHub:** https://github.com/danielchen05/isovita
- **PyPI:** https://pypi.org/project/isovita/

This repository preserves the analysis-specific code and workflows used in the manuscript.

## Repository contents

| Files | Description |
|---|---|
| `isovita.py` | IsoVita implementation used for manuscript analyses |
| `isovita_simulation.py` | Simulation utilities |
| `prepare_isovita_inputs.py` | Data preprocessing utilities |
| `DLPFC_IsoVita_run.ipynb` | DLPFC analysis |
| `HPC_IsoVita_run.ipynb` | Hippocampus analysis |
| `Figure1_*.ipynb` - `Figure5_*.ipynb` | Main manuscript figures |
| `SuppFigure_*.ipynb` | Supplementary figures |
| `results/` | Simulation and analysis outputs |
| `data/` | Input datasets, processed inputs, and analysis results (archived on Zenodo) |

## Data availability

The datasets, processed inputs, and analysis outputs are available on Zenodo:

**[10.5281/zenodo.23840334](https://doi.org/10.5281/zenodo.23840334)**

Download and extract the archive so that the `data/` directory is located at the repository root. See `data/README.md` for details.

## Reproducibility

The analyses were conducted in Python. To reproduce the manuscript results:

1. Clone this repository.
2. Download and extract the `data/` archive from Zenodo.
3. Install the required Python dependencies.
4. Run the relevant analysis and figure-generation notebooks.

The repository includes the analysis implementation used for the manuscript. For installation instructions and general usage of IsoVita, refer to the separate software repository.  

Python 3.10-3.12 is recommended.

Install the dependencies:

```bash
pip install -r requirements.txt
```

This repository contains the analysis code used to generate the results and figures reported in the IsoVita manuscript. The implementation is preserved independently of the maintained IsoVita Python package (https://github.com/danielchen05/isovita), which may include subsequent updates and improvements.

For reproduction of the manuscript results, use the scripts in this repository and the associated data and software environment.

## Questions

For questions about the manuscript analyses or figures, please open an issue in this repository. For software-related questions, please use the [IsoVita package repository](https://github.com/danielchen05/isovita).