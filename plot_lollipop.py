import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os


def plot_lollipop_multiple(df, target_dir):

    sns.set_theme(style="whitegrid")  # set style
    df = df.sort_values('voted')

    # df["change"] = df['sample_avg'] - df['LOOCV_orig'] 
    df = df.set_index('prompt')

    plt.figure(figsize=(6,15))
    y_range = np.arange(1, len(df.index) + 1)
    # y_range = df.index
    # colors = np.where(df['sample_avg'] > df['pred_deepseek'], '#a2f593', '#f7b0b0')
    # plt.hlines(y=y_range, xmin=df['pred_deepseek'], xmax=df['sample_avg'], color=colors, lw=5)
    # plt.scatter(df['voted'], y_range, color='#0e0f0f', s=30, label='Majority Baseline', zorder=3)
    plt.scatter(df['run_1'], y_range, color='#108577', s=30 , label='Scoring with LLM', zorder=3)
    plt.scatter(df['run_2'], y_range, color='#108577', s=30, zorder=3)
    plt.scatter(df['run_3'], y_range, color='#108577', s=30, zorder=3)
    plt.scatter(df['run_4'], y_range, color='#108577', s=30, zorder=3)
    plt.scatter(df['run_5'], y_range, color='#108577', s=30, zorder=3)
    # for (_, row), y in zip(df.iterrows(), y_range):
    #     plt.annotate(f"{row['change']:+.0%}", (max(row["LOOCV_orig"], row["sample_avg"]) + 4, y - 0.25))
    plt.legend(ncol=1, bbox_to_anchor=(0.5, 1.0), loc="lower center", frameon=False)

    plt.yticks(y_range, df.index, ha='left', fontsize=8)
    plt.tick_params(axis='y', which='major', pad=30)

    plt.xlabel('Macro F1')
    plt.xlim(0,1)
    plt.ylim(0, len(df)+1)

    plt.gcf().subplots_adjust(left=0.35)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(target_dir, 'compare_f1_llm_runs.pdf'))


def plot_lollipop_scatter(df, target_dir):

    sns.set_theme(style="whitegrid")  # set style
    df = df.sort_values('LOOCV_orig')

    # df["change"] = df['sample_avg'] - df['LOOCV_orig'] 
    df = df.set_index('prompt')

    plt.figure(figsize=(6,15))
    y_range = np.arange(1, len(df.index) + 1)
    # y_range = df.index
    colors = np.where(df['sample_avg'] > df['pred_deepseek'], '#a2f593', '#f7b0b0')
    plt.hlines(y=y_range, xmin=df['pred_deepseek'], xmax=df['sample_avg'], color=colors, lw=5)
    plt.scatter(df['f1_majority'], y_range, color='#0e0f0f', s=100, marker='|', label='Majority baseline', zorder=3)
    plt.scatter(df['sample_avg'], y_range, color='#108577', s=30 , label='Train on SRA-gen', zorder=3)
    plt.scatter(df['LOOCV_orig'], y_range, color='#1554b3', s=30, label='LOOCV on SRA', zorder=3)
    plt.scatter(df['pred_deepseek'], y_range, color='#02e3df', s=30, label='LLM scoring of SRA', zorder=3)
    # for (_, row), y in zip(df.iterrows(), y_range):
    #     plt.annotate(f"{row['change']:+.0%}", (max(row["LOOCV_orig"], row["sample_avg"]) + 4, y - 0.25))
    plt.legend(ncol=2, bbox_to_anchor=(0.5, 1.0), loc="lower center", frameon=False)

    plt.yticks(y_range, df.index, ha='left', fontsize=8)
    plt.tick_params(axis='y', which='major', pad=30)

    plt.xlabel('Macro F1')
    plt.xlim(0,1)
    plt.ylim(0, len(df)+1)

    plt.gcf().subplots_adjust(left=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, 'compare_f1_overall.pdf'))


def plot_lollipop_simple(results_file):

    sns.set_theme(style="whitegrid")  # set style

    df = pd.read_csv(results_file)
    df = df.sort_values('LOOCV_orig')

    df["change"] = df['sample_avg'] - df['LOOCV_orig'] 
    df = df.set_index('prompt')

    plt.figure(figsize=(6,15))
    y_range = np.arange(1, len(df.index) + 1)
    # y_range = df.index
    colors = np.where(df['sample_avg'] > df['LOOCV_orig'], '#a2f593', '#f7b0b0')
    plt.hlines(y=y_range, xmin=df['LOOCV_orig'], xmax=df['sample_avg'],
            color=colors, lw=5)
    plt.scatter(df['sample_avg'], y_range, color='#108577', s=30 , label='LLM Data', zorder=3)
    plt.scatter(df['LOOCV_orig'], y_range, color='#1554b3', s=30, label='Original Data', zorder=3)
    # for (_, row), y in zip(df.iterrows(), y_range):
    #     plt.annotate(f"{row['change']:+.0%}", (max(row["LOOCV_orig"], row["sample_avg"]) + 4, y - 0.25))
    plt.legend(ncol=2, bbox_to_anchor=(0.5, 1.0), loc="lower center", frameon=False)

    plt.yticks(y_range, df.index, ha='left', fontsize=8)
    plt.tick_params(axis='y', which='major', pad=30)

    plt.xlabel('Macro F1')
    plt.xlim(0,1)
    plt.ylim(0, len(df)+1)

    plt.gcf().subplots_adjust(left=0.35)
    plt.tight_layout()
    # plt.show()
    # plt.savefig('compare_same_dist_simple' + suffix + '.pdf')
    plt.savefig(results_file[:results_file.rindex('.')] + '_simple.pdf')


def plot_lollipop(results_file):

    sns.set_theme(style="whitegrid")  # set style

    df = pd.read_csv(results_file)
    df = df.sort_values('LOOCV_orig')

    df["sd_pos"] = df['sample_avg'] + df['sample_sd'] 
    df["sd_neg"] = df['sample_avg'] - df['sample_sd'] 
    df = df.set_index('prompt')

    plt.figure(figsize=(6,15))
    y_range = np.arange(1, len(df.index) + 1)
    # y_range = df.index
    # colors = np.where(df['sample_avg'] > df['LOOCV_orig'], '#d9d9d9', '#f7b0b0')
    plt.hlines(y=y_range, xmin=df['sd_pos'], xmax=df['sample_avg'], color='#c5e6e2', lw=3, label='LLM Data (SD)')
    plt.hlines(y=y_range, xmin=df['sd_neg'], xmax=df['sample_avg'], color='#c5e6e2', lw=3)
    plt.scatter(df['sample_min'], y_range, color='#b01f05', s=30 , label='LLM Data (min)', zorder=3)
    plt.scatter(df['sample_max'], y_range, color='#02a812', s=30 , label='LLM Data (max)', zorder=3)
    plt.scatter(df['sample_avg'], y_range, color='#108577', s=30 , label='LLM Data (avg)', zorder=3)
    plt.scatter(df['LOOCV_orig'], y_range, color='#1554b3', s=30, label='Original Data', zorder=3)
    # for (_, row), y in zip(df.iterrows(), y_range):
    #     plt.annotate(f"{row['change']:+.0%}", (max(row["LOOCV_orig"], row["sample_avg"]) + 4, y - 0.25))
    plt.legend(ncol=2, bbox_to_anchor=(0.5, 1.0), loc="lower center", frameon=False)

    plt.yticks(y_range, df.index, ha='left', fontsize=8)
    plt.tick_params(axis='y', which='major', pad=30)
    plt.xlim(0,1)
    plt.ylim(0, len(df)+1)

    plt.xlabel('Macro F1')

    plt.gcf().subplots_adjust(left=0.35)
    plt.tight_layout()
    # plt.show()
    plt.savefig(results_file[:results_file.rindex('.')] + '.pdf')


# plot_lollipop_simple(results_file='results_LR/same_distributionLR_2_way.csv')
# plot_lollipop(results_file='results_LR/same_distributionLR_2_way.csv')

# plot_lollipop_simple(results_file='results_LR/same_distributionLR_3_way.csv')
# plot_lollipop(results_file='results_LR/same_distributionLR_3_way.csv')

# plot_lollipop_simple(results_file='results_LR/same_distributionLR_5_way.csv')
# plot_lollipop(results_file='results_LR/same_distributionLR_5_way.csv')



# plot_lollipop_simple(results_file='results_clean_LR/same_distributionLR_clean_2_way.csv')
# plot_lollipop(results_file='results_clean_LR/same_distributionLR_clean_2_way.csv')

# plot_lollipop_simple(results_file='results_clean_LR/same_distributionLR_clean_3_way.csv')
# plot_lollipop(results_file='results_clean_LR/same_distributionLR_clean_3_way.csv')

# plot_lollipop_simple(results_file='results_clean_LR/same_distributionLR_clean_5_way.csv')
# plot_lollipop(results_file='results_clean_LR/same_distributionLR_clean_5_way.csv')


###
# Combine LLM Labels, LOOCV and models trained on synthetic data
df_majority = pd.read_csv('results/majority.csv')
df_llm_data = pd.read_csv('results_LR/same_distributionLR_5_way.csv')
df_llm_data.rename(columns={'LOOCV orig': 'LOOCV_orig'}, inplace=True)
# df_llm_pred = pd.read_csv('llm_preds_1/f1_scores.csv')
# df_llm_pred.rename(columns={'macro_f1': 'pred_deepseek'}, inplace=True)
df_llm_pred = pd.read_csv('results/llm_scoring_with_majority.csv')
df_llm_pred['pred_deepseek'] = (df_llm_pred['run_1'] + df_llm_pred['run_2'] + df_llm_pred['run_3'] + df_llm_pred['run_4'] + df_llm_pred['run_5'])/5
df = pd.merge(left=df_llm_data, right=df_llm_pred, left_on='prompt', right_on='prompt')
df = pd.merge(left=df, right=df_majority, left_on='prompt', right_on='prompt')
print(df[['pred_deepseek', 'LOOCV_orig', 'sample_avg', 'f1_majority']].mean())
plot_lollipop_scatter(df=df, target_dir='results')


# df_full = None
# for run in range(1, 6):

#     df_results = pd.read_csv('llm_preds_' + str(run) + '/f1_scores.csv')
#     df_results.columns = ['prompt', 'llm_preds_' + str(run)]

#     if df_full is None:
#         df_full  =df_results
    
#     else:
#         df_full=pd.merge(left=df_full, right=df_results, left_on='prompt', right_on='prompt')
    
# print(df_full)
# plot_lollipop_multiple(df=df_full)

# df = pd.read_csv('results/llm_scoring_with_majority.csv')
# plot_lollipop_multiple(df=df, target_dir='results')