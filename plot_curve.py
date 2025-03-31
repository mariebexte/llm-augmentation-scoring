import pandas as pd
import matplotlib.pyplot as plt

import os
import sys


def get_max_size(columns):

    last = columns[-1]
    return int(last[:last.index('_')])


def plot_curve(results_file, result_dir):

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    df = pd.read_csv(results_file)
    df = df.set_index('prompt')
    max_size = get_max_size(df.columns)
    # df = df.drop(columns=['prompt'])
    df.loc['mean'] = df.mean()
    df['prompt_name'] = df.index

    # TODO: Must dynamically determine this!
    sizes = list(range(1, max_size+1))
    # sizes = list(range(1, 28))

    for prompt_name, df_prompt in df.groupby('prompt_name'):

        avg = []
        min = []
        max = []

        for size in sizes:

            avg.append(df_prompt.loc[:, str(size) + '_avg'].iloc[0])
            min.append(df_prompt.loc[:, str(size) + '_min'].iloc[0])
            max.append(df_prompt.loc[:, str(size) + '_max'].iloc[0])

        plt.plot(sizes, min, color='r', label='worst')    
        plt.plot(sizes, avg, color='b', label='avg')    
        plt.plot(sizes, max, color='green', label='best') 

        plt.title(prompt_name)

        plt.legend(ncol=3)

        plt.xlim(1, 100)
        plt.xlabel('# samples per label')

        plt.ylim(0,1)
        plt.ylabel('Macro F1')

        plt.savefig(os.path.join(result_dir, str(prompt_name) + '.pdf'))  

        plt.cla()
    

# plot_curve(results_file='results_LR/llm_balancedLR_5_way.csv', result_dir='curves_5_way')
# plot_curve(results_file='results_LR/llm_balancedLR_3_way.csv', result_dir='curves_3_way')
# plot_curve(results_file='results_LR/llm_balancedLR_2_way.csv', result_dir='curves_2_way')

plot_curve(results_file='results_clean_LR/llm_balancedLR_clean_5_way.csv', result_dir='curves_clean_5_way')
plot_curve(results_file='results_clean_LR/llm_balancedLR_clean_3_way.csv', result_dir='curves_clean_3_way')
plot_curve(results_file='results_clean_LR/llm_balancedLR_clean_2_way.csv', result_dir='curves_clean_2_way')