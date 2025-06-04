import pandas as pd
import matplotlib.pyplot as plt

import os
import sys


def get_max_size(columns):

    last = columns[-1]

    if 'max_' in last:
        return int(last[last.index('_')+1:last.rindex('_')])
    
    else:
        return int(last[:last.index('_')])


def plot_curve(results_file, result_dir, prefix='', sizes=None):

    plt.rcParams["figure.figsize"] = (6,7)
    # plt.figure(figsize=(5,8))
    # plt.figure(figsize=(9,3))

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    df = pd.read_csv(results_file)
    df = df.set_index('prompt')
    max_size = get_max_size(df.columns)
    # df = df.drop(columns=['prompt'])
    df.loc['mean'] = df.mean()
    df['prompt_name'] = df.index

    if sizes is None:
        sizes = list(range(1, max_size+1))
    # sizes = list(range(1, 28))

    for prompt_name, df_prompt in df.groupby('prompt_name'):

        avg = []
        min = []
        max = []

        for size in sizes:

            avg.append(df_prompt.loc[:, prefix + str(size) + '_avg'].iloc[0])
            min.append(df_prompt.loc[:, prefix + str(size) + '_min'].iloc[0])
            max.append(df_prompt.loc[:, prefix + str(size) + '_max'].iloc[0])

        plt.plot(sizes, min, color='r', label='worst')    
        plt.plot(sizes, avg, color='b', label='avg')    
        plt.plot(sizes, max, color='green', label='best') 

        plt.title(prompt_name, fontsize=24)

        # plt.legend(ncol=3, fontsize=20, loc='upper center')
        plt.legend(ncol=1, fontsize=22, loc='upper left')
        plt.xticks(sizes)
        plt.xlim(4, 26)
        # plt.xlim(0,101)
        plt.xlabel('# samples per label', fontsize=24)

        plt.tick_params(axis='both', which='major', labelsize=17)

        plt.ylim(0,1)
        plt.ylabel('Macro F1', fontsize=24)

        plt.savefig(os.path.join(result_dir, str(prompt_name) + prefix + '.pdf'), bbox_inches="tight")  

        plt.cla()


def plot_curve_all_methods(results_file, result_dir, prefix='', sizes=None):

    plt.rcParams["figure.figsize"] = (10,6)
    # plt.figure(figsize=(5,8))
    # plt.figure(figsize=(9,3))

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    df = pd.read_csv(results_file)
    df = df.set_index('prompt')
    max_size = get_max_size(df.columns)
    # df = df.drop(columns=['prompt'])
    df.loc['mean'] = df.mean()
    df['prompt_name'] = df.index

    if sizes is None:
        sizes = list(range(1, max_size+1))
    # sizes = list(range(1, 28))

    for prompt_name, df_prompt in df.groupby('prompt_name'):

        avg = []
        min = []
        max = []

        for size in sizes:

            avg.append(df_prompt.loc[:, prefix + str(size) + '_avg'].iloc[0])
            min.append(df_prompt.loc[:, prefix + str(size) + '_min'].iloc[0])
            max.append(df_prompt.loc[:, prefix + str(size) + '_max'].iloc[0])
        
        if prompt_name == 'mean':

            plt.plot(sizes, min, color='r', label='worst')    
            plt.plot(sizes, avg, color='b', label='avg')    
            plt.plot(sizes, max, color='green', label='best') 

        else:

            plt.plot(sizes, avg, color='b', alpha=0.05)

    # plt.title(prompt_name, fontsize=24)

    # plt.legend(ncol=3, fontsize=20, loc='upper center')
    plt.legend(ncol=1, fontsize=18, loc='upper left')
    # plt.xticks(sizes)
    # plt.xlim(4, 26)
    plt.xlim(0,101)
    plt.xlabel('# samples per label', fontsize=18)

    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.ylim(0,1)
    plt.ylabel('Macro F1', fontsize=18)

    plt.savefig(os.path.join(result_dir, 'all_prompts' + prefix + '.pdf'), bbox_inches="tight")  

    plt.cla()
    

# plot_curve(results_file='results_LR/llm_balancedLR_5_way.csv', result_dir='curves_5_way')
# plot_curve(results_file='results_LR/llm_balancedLR_3_way.csv', result_dir='curves_3_way')
# plot_curve(results_file='results_LR/llm_balancedLR_2_way.csv', result_dir='curves_2_way')

# plot_curve(results_file='results_clean_LR/llm_balancedLR_clean_5_way.csv', result_dir='curves_clean_5_way')
# plot_curve(results_file='results_clean_LR/llm_balancedLR_clean_3_way.csv', result_dir='curves_clean_3_way')
# plot_curve(results_file='results_clean_LR/llm_balancedLR_clean_2_way.csv', result_dir='curves_clean_2_way')

plot_curve_all_methods(results_file='results_pretrained_balanced_all-MiniLM-L6-v2/llm_balancedall-MiniLM-L6-v2_5_way.csv', result_dir='curves_pretrained_all-MiniLM-L6-v2', sizes=[1,2,3,5,10,25,50,100])
plot_curve_all_methods(results_file='results_pretrained_balanced_all-MiniLM-L6-v2/llm_balancedall-MiniLM-L6-v2_5_way.csv', result_dir='curves_pretrained_all-MiniLM-L6-v2', prefix='max_', sizes=[1,2,3,5,10,25,50,100])
# plot_curve(results_file='results_pretrained_balanced_all-MiniLM-L6-v2/llm_balancedall-MiniLM-L6-v2_5_way.csv', result_dir='curves_pretrained_all-MiniLM-L6-v2', sizes=[1,2,3,5,10,25,50,100])
# plot_curve(results_file='results_pretrained_balanced_all-MiniLM-L6-v2/llm_balancedall-MiniLM-L6-v2_5_way.csv', result_dir='curves_pretrained_all-MiniLM-L6-v2', prefix='max_', sizes=[1,2,3,5,10,25,50,100])

# plot_curve(results_file='results_clean_balanced_all-MiniLM-L6-v2/llm_balancedall-MiniLM-L6-v2_clean_5_way.csv', result_dir='results_clean_balanced_all-MiniLM-L6-v2', prefix='', sizes=[2,3,4,5,6,7,14,28])
# plot_curve(results_file='results_clean_balanced_all-MiniLM-L6-v2/llm_balancedall-MiniLM-L6-v2_clean_5_way.csv', result_dir='results_clean_balanced_all-MiniLM-L6-v2', prefix='max_', sizes=[2,3,4,5,6,7,14,28])

# plot_curve(results_file='results_CURVES_clean_balanced_all-MiniLM-L6-v2/llm_balancedall-MiniLM-L6-v2_clean_5_way.csv', result_dir='results_CURVES_clean_balanced_all-MiniLM-L6-v2', prefix='', sizes=[5,10,15,20,25])
# plot_curve(results_file='results_CURVES_clean_balanced_all-MiniLM-L6-v2/llm_balancedall-MiniLM-L6-v2_clean_5_way.csv', result_dir='results_CURVES_clean_balanced_all-MiniLM-L6-v2', prefix='max_', sizes=[5,10,15,20,25])
# plot_curve(results_file='results_CURVES_dirty_balanced_all-MiniLM-L6-v2/llm_balancedall-MiniLM-L6-v2_clean_5_way.csv', result_dir='results_CURVES_dirty_balanced_all-MiniLM-L6-v2', prefix='', sizes=[5,10,15,20,25])
# plot_curve(results_file='results_CURVES_dirty_balanced_all-MiniLM-L6-v2/llm_balancedall-MiniLM-L6-v2_clean_5_way.csv', result_dir='results_CURVES_dirty_balanced_all-MiniLM-L6-v2', prefix='max_', sizes=[5,10,15,20,25])