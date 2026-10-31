# Data

This folder contains the input datasets, processed data, and analysis results used in the IsoVita manuscript.

## Directory structure

```text
data/
├── chess3.1.3.mapfile.tsv
├── DLPFC/
│   ├── metadata.tsv
│   ├── tx2gene.csv
│   ├── tximport_counts.txt
│   ├── isovita_inputs/
│   └── isovita_results/
└── HPC/
    ├── metadata.tsv
    ├── tx2gene.csv
    ├── HPC_transcript_counts.h5ad
    ├── isovita_inputs/
    └── isovita_results/
```

- **`chess3.1.3.mapfile.tsv`**: CHESS transcript-to-gene annotation file.
- **`DLPFC/`**: Dorsolateral prefrontal cortex transcript expression data, sample metadata, processed IsoVita inputs, and analysis results.
- **`HPC/`**: Hippocampus transcript expression data, sample metadata, processed IsoVita inputs, and analysis results.

Within each dataset, `isovita_inputs/` contains processed inputs used by IsoVita, while `isovita_results/` contains outputs from the analysis.

## Usage

Download the data archive from Zenodo and extract it so that the `data/` directory is located at the root of the [IsoVita-paper repository](https://github.com/danielchen05/IsoVita-paper).

The analysis and figure-generation notebooks in the repository use these files to reproduce the manuscript results.

## Data availability

The complete data directory is archived on Zenodo.

The IsoVita software package is maintained separately at [isovita](https://github.com/danielchen05/isovita).