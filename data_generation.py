from ollama import Client
from prompts import CORRECT, PARTIALLY, CONTRADICTORY, IRRELEVANT, NON_DOMAIN
from tqdm import tqdm

import pandas as pd
import sys
import os


# HOST = 'http://catalpa-llm.fernuni-hagen.de:11434'
HOST = 'localhost:11434'

target_folder = 'generated_llm_answers'

if not os.path.exists(target_folder):
    os.mkdir(target_folder)


def invoke_llm(prompt):
    client = Client(host=HOST)
    
    response = client.chat(model="deepseek-v2:latest", messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])

    return response['message']['content']


# Strip away enumeration
def clean_answer_df(df):

    df['text_clean'] = df['answer'].str.strip(' ')
    df['text_clean'] = df['text_clean'].str.strip('\n')
    df['text_clean'] = df['text_clean'].str.strip('-')
    df['text_clean'] = df['text_clean'].str.strip('1.')
    df['text_clean'] = df['text_clean'].str.strip('2.')
    df['text_clean'] = df['text_clean'].str.strip('3.')
    df['text_clean'] = df['text_clean'].str.strip('4.')
    df['text_clean'] = df['text_clean'].str.strip('5.')
    df['text_clean'] = df['text_clean'].str.strip('6.')
    df['text_clean'] = df['text_clean'].str.strip('7.')
    df['text_clean'] = df['text_clean'].str.strip('8.')
    df['text_clean'] = df['text_clean'].str.strip('9.')
    df['text_clean'] = df['text_clean'].str.strip('10.')
    df['text_clean'] = df['text_clean'].str.strip('*')
    df['text_clean'] = df['text_clean'].str.strip('**')
    df['text_clean'] = df['text_clean'].str.strip(' ')
    df['text_clean'] = df['text_clean'].str.strip('\"')
    df['text_clean'] = df['text_clean'].str.strip('**')
    df['text_clean'] = df['text_clean'].str.strip(' ')

    df = df[df['text_clean'].str.len() > 0]

    return df


# Remove answers that are regugitations of the prompt
def drop_prompt_artifacts(df, prompt):

    # Remove lines with xlm 
    df_clean = df.loc[~ df['answer'].astype(str).str.match(r'.*<.*>.*')]

    prompt_elements = prompt.split('\n')
    prompt_elements = [prompt_element for prompt_element in prompt_elements if prompt_element != '']

    prompt_elements.append('10 possible answers')
    prompt_elements.append('list of possible answers')
    prompt_elements.append('Here are 10')
    prompt_elements.append('Here are ten')
    prompt_elements.append('maximum of 20 words')
    prompt_elements.append('ten different responses')
    prompt_elements.append('10 different responses')

    for prompt_element in prompt_elements:

        df_clean = df_clean.loc[~ df['answer'].astype(str).str.contains(prompt_element, regex=False)]

    df_clean = df_clean.dropna()
    return df_clean


def get_answers(prompt, question, num_runs=6):

    answers = ''

    for i in range(num_runs):
        answers = answers + invoke_llm(prompt.format(question=question))

    df = pd.DataFrame({'answer': answers.split('\n')})
    df = clean_answer_df(df)
    df = drop_prompt_artifacts(df, prompt.format(question=question))
    df = df.dropna()

    return df


## Start generation for all prompts of interest
df_questions = pd.read_csv('data/prompts_without_pictures.csv')
question_ids = df_questions['id']
question_texts = df_questions['q_text']

for question_id, question_text in tqdm(zip(question_ids, question_texts), total=len(question_ids)):

    target_filename = os.path.join(target_folder, question_id + '_llm.csv')

    if not os.path.exists(target_filename):

        dfs = []

        for prompt in [(1.0, CORRECT), ('partially_correct_incomplete', PARTIALLY), ('contradictory', CONTRADICTORY), ('irrelevant', IRRELEVANT), ('non_domain', NON_DOMAIN)]:

            df_answers = pd.DataFrame()

            while len(df_answers) < 50:

                df_answers = get_answers(prompt=prompt[1], question=question_text)
                df_answers['label'] = prompt[0]
                df_answers['question'] = question_id
                df_answers['question_text'] = question_text
                
            df_answers = df_answers.head(50)
            dfs.append(df_answers)

        df_question = pd.concat(dfs).reset_index()
        df_question['id'] = df_question.index
        df_question['id'] = df_question['id'].apply(lambda x: 'deepseek_' + question_id + '_' + str(x))

        df_question = df_question[['id', 'question', 'question_text', 'answer', 'text_clean', 'label']]
        # Shuffle
        df_question = df_question.sample(frac=1)
        df_question.to_csv(os.path.join(target_filename), index=None)

        print('Done', question_id, len(df_question))
        print(df_question['label'].value_counts())
    
    else:

        print('Skipping, because it already ran:', question_id)
