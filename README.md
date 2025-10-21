# Biomedical Relation Extractor

Python tool for extracting biomedical relationships from text as `(entity1, relation, entity2)` triplets using SpaCy.

## Features

- Named Entity Recognition (NER) for biomedical entities.
- Relation extraction using regex patterns and dependency parsing.
- Triplet filtering to remove duplicates and contradictory relations.
- GPU support for faster processing.
- Batch processing for large datasets.

## Requirements

```bash
pip install spacy cupy
# Optional: install biomedical NER model
python -m spacy download en_ner_bionlp13cg_md
python -m spacy download en_core_web_sm
