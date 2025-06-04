# Generating Training Data for Content Scoring

This repository contains data and experiment code for our 2025 BEA paper "Is Lunch Free Yet? Overcoming the Cold-Start Problem in Supervised Content Scoring using Zero-Shot LLM-Generated Training Data"

<img src="https://github.com/user-attachments/assets/ef9ee002-34a6-41b9-86c6-32b4a9f74acf" width=550>

## Data
We include a copy of the original SRA data in `data/SRA_SEB`.
We also provide our LLM-generated SRA-gen data in `data/generated_llm_answers` and `data/generated_llm_answers_2`.
The two sets of generated answers each contain 250 answers per prompt (50 for each of the five labels in the dataset).
Answers were generated with deepseek-v2 using the `data_generation.py` script. The prompts it uses can be found in `prompts.py`.

For the three prompts `ME_27b`, `PS_4bp` and `VB_1`, the file in `generated_llm_answers` contains an additional column `label_clean` with manual labels that were adjudicated between three annotators.
For the inividual annotations before adjudication, see the files in `kappa`:
- `kappa/*trial*` files contain our re-annotation of the original data.
- `kappa/*deepseek*` files contain our annotation of the generated data.

To compute data statistics (length, token overlap, TTR), run `calculate_answer_statistics.py`.

## Experiments

### Baselines
`baseline_majority.py`, `baseline_llm_scoring.py` for direct scoring of the SRA_SEB data with deepseek.

### Training on original SRA_SEB vs. our as-generated SRA-gen data, controlling for identical label distribution (Section 6.1)
Aggregated performance (Table 4)
- Regression and pretrained SBERT model: `compare_same_distribution.py`
- Finetuned BERT and SBERT model: `compare_same_distribution_deep.py`

Performance spread for multiple samples, prompt-wise (Figures 2, 7, 8, 9)
- Plot with `plot_lollipop.py`

### Influence of the amount of SRA-gen data used as training data (Section 6.2 / Figure 10)
- Plot with `plot_curve.py`


### Training on as-generated vs. manually cleaned SRA-gen data (Section 7.2 / Table 8)
40 answers per prompt, following the same label distribution as in the original data:
- Regression and pretrained SBERT model: `compare_same_distribution_clean.py`
- Finetuned BERT and SBERT model: `compare_same_distribution_clean_deep.py`

Full set of 250 answers per prompt:
- Regression model: `compare_data_quality_LR.py`
- Pretrained SBERT model: `compare_data_quality_pretrained.py`
- Finetuned BERT and SBERT model: `compare_data_quality_deep.py`

### Training on balanced sample of as-generated vs. manually cleaned SRA-gen data (Section 7.2 / Figure 4)
`compare_balanced_clean_deep.py` runs the learning curve experiment for the SBERT model.
`postprocess_curve_data.py` derives a dataframe that can then be used to plot the learning curves with `plot_curve.py`.

