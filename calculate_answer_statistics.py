import pandas as pd
import os
import sys
import spacy
import regex

from taaled import ld
from pylats import lats

from nltk.probability import FreqDist

prompt_file = 'data/prompts_without_pictures.csv'
df_prompts = pd.read_csv(prompt_file)

orig_data_path = 'data/SRA_SEB'
gen_data_path_1 = 'data/generated_llm_answers'
gen_data_path_2 = 'data/generated_llm_answers_2'

labels = ['1.0', 'partially_correct_incomplete', 'contradictory', 'non_domain', 'irrelevant']
nlp = spacy.load("en_core_web_sm")


def get_avg_token_len(row):

    doc = nlp(row)
    token_lens = [len(token.text) for token in doc if not token.pos_ == 'PUNCT']
    if len(token_lens) > 0:
        avg_token_len = sum(token_lens)/len(token_lens)
    else:
        return 0

    return avg_token_len


def get_ttr(answer_list):

    full_string = ' '.join(answer_list)
    clnsmpl = lats.Normalize(full_string, lats.ld_params_en)
    ldvals = ld.lexdiv(clnsmpl.toks)
    values = {'MATTR': ldvals.mattr, 'MTLD': ldvals.mtld, 'TTR': ldvals.ttr}
    
    return values


# Write length stats about a datset to file
def calculate_stats(list_of_data_sources, target_column, answer_column, result_file, file_prefix='', file_suffix=''):

    stats = {}

    for prompt in df_prompts.id:

        print(prompt)

        dfs = []
        for source in list_of_data_sources:

            if file_suffix.endswith('.tsv'):
                dfs.append(pd.read_csv(os.path.join(source, file_prefix+prompt+file_suffix), sep='\t'))
            else:
                dfs.append(pd.read_csv(os.path.join(source, file_prefix+prompt+file_suffix)))
        
        df = pd.concat(dfs)

        prompt_stats = {}

        label_dist = dict(df[target_column].value_counts())
        for label in labels:
            prompt_stats[label] = label_dist.get(label, 0)

        if len(df) != sum(prompt_stats.values()):
            print('Individual answers not adding up!')
            sys.exit(0)

        prompt_stats['num_answers'] = len(df)

        df['answer_length_in_chars'] = df[answer_column].apply(lambda entry: len(entry))
        prompt_stats['avg_answer_len_in_chars'] = df['answer_length_in_chars'].mean()

        df['token_length_in_chars'] = df[answer_column].apply(get_avg_token_len)
        prompt_stats['avg_token_len_in_chars'] = df['token_length_in_chars'].mean()

        ttr_values = get_ttr(df[answer_column].tolist())

        for key, value in ttr_values.items():
            prompt_stats[key] = value

        stats[prompt] = prompt_stats

        df_stats = pd.DataFrame.from_dict(stats).T
        df_stats.index.name = 'prompt'
        df_stats.loc['mean'] = df_stats.mean()
        df_stats.to_csv(result_file)


def get_token_set(answer_list):

    full_answer_string = ' '.join(answer_list).lower()
    all_tokens = nlp(full_answer_string)
    all_tokens = [token.text for token in all_tokens if not token.pos_ == 'PUNCT']
    return set(all_tokens)


def get_token_fd(answer_list):

    full_answer_string = ' '.join(answer_list).lower()
    all_tokens = nlp(full_answer_string)
    all_tokens = [token.text for token in all_tokens if not token.pos_ == 'PUNCT']
    return FreqDist(all_tokens)


# Write stats about tokens in the two datasets
def compare_corpora(sources_orig, sources_llm, result_file):

    stats = {}

    for prompt in df_prompts.id:

        dfs_orig = []
        for source in sources_orig:
            dfs_orig.append(pd.read_csv(os.path.join(source, 'SRA_allAnswers_prompt'+prompt+'.tsv'), sep='\t'))
        df_orig = pd.concat(dfs_orig)

        dfs_llm = []
        for source in sources_llm:
            dfs_llm.append(pd.read_csv(os.path.join(source, prompt+'_llm.csv')))
        df_llm = pd.concat(dfs_llm)

        prompt_stats = {}

        tokens_orig = get_token_set(df_orig['AnswerText'])
        tokens_llm = get_token_set(df_llm['text_clean'])

        stats[prompt] = {
            'overlap': len(tokens_orig.intersection(tokens_llm)),
            'unique_orig': len(tokens_orig.difference(tokens_llm)),
            'unique_llm': len(tokens_llm.difference(tokens_orig)),
            'num_orig': len(tokens_orig),
            'num_llm': len(tokens_llm),
        }

        df_stats = pd.DataFrame.from_dict(stats).T
        df_stats.index.name = 'prompt'
        df_stats.loc['mean'] = df_stats.mean()
        df_stats.to_csv(result_file)


# To compare the two data sources wrt the types present in the answers
def extract_type_overview(sources_orig, sources_llm, result_dir):

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    for prompt in df_prompts.id:

        prompt_result_dir = os.path.join(result_dir, prompt)

        if not os.path.exists(prompt_result_dir):
            os.mkdir(prompt_result_dir)

        dfs_orig = []
        for source in sources_orig:
            dfs_orig.append(pd.read_csv(os.path.join(source, 'SRA_allAnswers_prompt'+prompt+'.tsv'), sep='\t'))
        df_orig = pd.concat(dfs_orig)

        dfs_llm = []
        for source in sources_llm:
            dfs_llm.append(pd.read_csv(os.path.join(source, prompt+'_llm.csv')))
        df_llm = pd.concat(dfs_llm)

        tokens_orig = get_token_set(df_orig['AnswerText'])
        tokens_llm = get_token_set(df_llm['text_clean'])

        shared_tokens = tokens_orig.intersection(tokens_llm)
        unique_sra = tokens_orig.difference(tokens_llm)
        unique_llm = tokens_llm.difference(tokens_orig)

        most_frequent_sra = get_token_fd(df_orig['AnswerText'])
        most_frequent_llm = get_token_fd(df_llm['text_clean'])

        with open(os.path.join(prompt_result_dir, 'shared_types.txt'), 'w') as file:

            for type in shared_tokens:

                file.write(type+'\n')

        with open(os.path.join(prompt_result_dir, 'unique_types_sra.txt'), 'w') as file:

            for type in unique_sra:

                file.write(str(most_frequent_sra[type])+'\t'+type+'\n')

        with open(os.path.join(prompt_result_dir, 'unique_types_llm.txt'), 'w') as file:

            for type in unique_llm:

                file.write(str(most_frequent_llm[type])+'\t'+type+'\n')

        with open(os.path.join(prompt_result_dir, 'all_types_sra.txt'), 'w') as file:

            for type, count in most_frequent_sra.most_common():

                file.write(str(count)+'\t'+type+'\n')

        with open(os.path.join(prompt_result_dir, 'all_types_llm.txt'), 'w') as file:

            for type, count in most_frequent_llm.most_common():

                file.write(str(count)+'\t'+type+'\n')


# Print answers that contain Chinese characters
def find_chinese(sources_llm):

    stats = {}

    for prompt in df_prompts.id:

        dfs_llm = []
        for source in sources_llm:
            dfs_llm.append(pd.read_csv(os.path.join(source, prompt+'_llm.csv')))
        df_llm = pd.concat(dfs_llm)

        for answer in df_llm['text_clean']:
            result = regex.findall(r'\p{Han}+', answer)

            if len(result) > 0:
                print(answer, result)


calculate_stats(list_of_data_sources=[orig_data_path], file_prefix='SRA_allAnswers_prompt', file_suffix='.tsv', target_column='Score', answer_column='AnswerText', result_file='data/stats_orig.csv')
calculate_stats(list_of_data_sources=[gen_data_path_1, gen_data_path_2], file_suffix='_llm.csv', target_column='label', answer_column='text_clean', result_file='data/stats_llm.csv')

# compare_corpora(sources_orig=[orig_data_path], sources_llm=[gen_data_path_1, gen_data_path_2], result_file='data/compare_token_sets.csv')

# find_chinese(sources_llm=[gen_data_path_1, gen_data_path_2])
# extract_type_overview(sources_orig=[orig_data_path], sources_llm=[gen_data_path_1, gen_data_path_2], result_dir='type_analysis')