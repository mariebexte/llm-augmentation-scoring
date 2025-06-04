import pandas as pd
from ollama import Client
from tqdm import tqdm
from copy import deepcopy
from metrics import calculate_macro_f1

import sys
import os


HOST = 'localhost:11434'

SCORING_PROMPT = """
<purpose>
You are a school teacher.
A student has answered the following question: 
{question}
This is the answer the student gave:
{answer}
You now have to score this answer.
These are the possible scores:
Correct: A correct answer to the question.
Partially correct or incomplete: This means that the student answer is a partially correct answer that contains some but not all necessary information.
Contradictory: This means that the student answer is not correct and explicitly contradicts the correct answer.
Irrelevant: This means that the student answer is talking about domain content but not providing the necessary information to be correct.
Non-domain: This means that the student answer does not include domain content, e.g., "I don't know", "what the book says", "you are stupid".
</purpose>
<format_rules>
Only output the score.
</format_rules>
<output>
Decide on the score of the student answer.
</output>
"""


score_map = {
    'correct': 1.0,
    'partially correct or incomplete': 'partially_correct_incomplete',
    'contradictory': 'contradictory',
    'irrelevant': 'irrelevant',
    'non-domain': 'non_domain'
}


ref_answers = {
    'PS_4bp': 'When the string is longer, the pitch will be lower.',
    'VB_1': 'An experiment is controlled if only one variable is changed at a time.',
    'ME_27b': 'If the key sticks, the key is iron; if the key does not stick the key is aluminum.'
}


def ask_llm_for_prediction(prompt):

    client = Client(host=HOST)
    
    response = client.chat(model="deepseek-v2:latest", messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])

    answer = response['message']['content']
    return answer


## For each instance, obtain prediction from LLM
def score_prompt(df, answer_column, target_column, question, ref_answer, target_path):

    texts = df[answer_column]
    preds = []

    for answer in tqdm(texts, total=len(texts)):

        prompt = SCORING_PROMPT.format(question=question, answer=answer, ref_answer=ref_answer)

        pred_raw='nothing'
        while pred_raw not in score_map.keys():
            pred_raw = ask_llm_for_prediction(prompt)
            pred_raw = pred_raw.strip(' ')
            pred_raw = pred_raw.strip('-')
            pred_extracted = pred_raw.strip(' ')
            # pred_extracted = pred_raw[:pred_raw.index(':')]
            pred_extracted = pred_extracted.lower()
            if pred_extracted not in score_map.keys():
                print(pred_extracted, 'not in labels, retrying...')
            else:
                print('PRED', pred_extracted)

        pred = score_map[pred_extracted]
        preds.append(pred)
    
    df_copy = deepcopy(df)
    df_copy['pred_deepseek'] = preds
    df_copy.to_csv(target_path)

    return calculate_macro_f1(y_true=df[target_column], y_pred=preds)



for run in range(1, 6):
    results_dir = 'results_llm_preds_' + str(run)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Go over each prompt
    df_questions = pd.read_csv('data/prompts_without_pictures.csv').set_index('id')
    data_source = 'data/SRA_SEB'
    prompt_performances = {}

    for prompt in os.listdir(data_source):

        if 'prompt' in prompt:

            prompt_name = prompt[prompt.index('prompt')+len('prompt'):prompt.index('.')]

            if prompt_name in df_questions.index:
                question = df_questions.loc[prompt_name]

                print(prompt_name, question)

                df = pd.read_csv(os.path.join(data_source, prompt), sep='\t')
                prompt_f1 = score_prompt(df=df, answer_column='AnswerText', target_column='Score', question=question, target_path=os.path.join(results_dir, prompt_name + '_pred_deepseek.csv'))

                prompt_performances[prompt_name] = prompt_f1

                df_performances = pd.DataFrame.from_dict(prompt_performances, orient='index')
                df_performances.columns = ['macro_f1'] 
                df_performances.index.name = 'prompt' 
                df_performances.to_csv(os.path.join(results_dir, 'f1_scores.csv'))



# def get_majority(row):

#     dict_votes = dict(row.value_counts())
#     max_vote = max(dict_votes, key=dict_votes.get)
#     return max_vote


# def get_voted_results():

#     df_prompts = pd.read_csv('data/prompts_without_pictures.csv')
#     prompt_performances = {}

#     for prompt in df_prompts['id']:

#         # Merge predictions from all runs
#         df_full = None
#         for run in range(1, 6):

#             df = pd.read_csv(os.path.join('results_llm_preds_' + str(run), prompt + '_pred_deepseek.csv'))
#             df.rename(columns={'pred_deepseek': 'pred_deepseek_' + str(run)}, inplace=True)
            
#             if df_full is None:
#                 df_full = df
            
#             else:
#                 df = df[['AnswerId', 'pred_deepseek_' + str(run)]]
#                 df_full = pd.merge(left=df_full, right=df, left_on='AnswerId', right_on='AnswerId')
        
#         pred_columns = ['pred_deepseek_' + str(run) for run in range(1, 6)]
#         df_full['pred_deepseek_voted'] = df_full[pred_columns].apply(get_majority, axis=1)
#         print(df_full)

#         prompt_performances[prompt] = {'voted': calculate_macro_f1(y_true=df_full['Score'], y_pred=df_full['pred_deepseek_voted']),
#                                        'run_1': calculate_macro_f1(y_true=df_full['Score'], y_pred=df_full['pred_deepseek_1']),
#                                        'run_2': calculate_macro_f1(y_true=df_full['Score'], y_pred=df_full['pred_deepseek_2']),
#                                        'run_3': calculate_macro_f1(y_true=df_full['Score'], y_pred=df_full['pred_deepseek_3']),
#                                        'run_4': calculate_macro_f1(y_true=df_full['Score'], y_pred=df_full['pred_deepseek_4']),
#                                        'run_5': calculate_macro_f1(y_true=df_full['Score'], y_pred=df_full['pred_deepseek_5']),
#                                        }
#     df_overall = pd.DataFrame.from_dict(prompt_performances).T
#     df_overall.index.name='prompt'
#     df_overall.to_csv('results/llm_scoring_with_majority.csv')
#     print(df_overall)

# get_voted_results()

