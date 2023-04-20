import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import openai
import pandas as pd
import numpy as np
rand = np.random

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    RetryError,
) # for exponential backoff

rand.seed(100)

DIR = Path('table_data') / 'sato_tables' / 'all'
INFILE = sys.argv[1]
MODEL = sys.argv[2]

LIMIT_ROWS = 10

MODELS = {
    'ada': 'ada:ft-university-of-chicago-2023-04-13-22-30-17',
    'curie': 'curie:ft-university-of-chicago-2023-04-13-22-35-20',
}

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_response(prompt, model):
    '''
    send the prompt and data to chatgpt.
    If we do not get the answer in timeout seconds,
    resend the request.
    ...I have a feeling this will not work...
    '''
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=0,
        max_tokens=30,
        stop='\n',
    )
    return response["choices"][0]["text"]

def parse_response(response):
    return response.strip().split(', ')

def best_matches(arr1, arr2):
    if min(len(arr1), len(arr2)) == 0:
        return []
    if arr2[0].startswith(arr1[0]):
        matches = best_matches(arr1[1:], arr2[1:])
        return [(0, 0)] + [(m1+1, m2+1) for m1, m2 in matches]
    else:
        matches1 = best_matches(arr1, arr2[1:])
        matches1 = [(m1, m2+1) for m1, m2 in matches1]
        matches2 = best_matches(arr1[1:], arr2)
        matches2 = [(m1+1, m2) for m1, m2 in matches2]
        if len(matches2) > len(matches1):
            return matches2
        return matches1
    

df = pd.read_csv(INFILE)
outcol = f"label_pred_{MODEL}"

files = []
cols = []
good = 0
bad = 0
bad_resp = 0
bad_length = 0
done = 0
print(len(df['true_file'].unique()))
for fname, subset in df.groupby('true_file'):
    print(good, bad, bad_resp, bad_length, done)
    if not all(subset[outcol].isna()):
        done += 1
        continue
    subset = subset.sort_values('col')
    fdf = pd.read_csv(DIR / fname)
    num_cols = min(len(fdf.columns), len(subset))
    best_match_idxs = best_matches(list(subset['label']), list(fdf.columns))
    subset_idxs = [i1 for i1, i2 in best_match_idxs]
    fdf_idxs = [i2 for i1, i2 in best_match_idxs]
    if len(best_match_idxs) < num_cols:
        print("COL MISMATCH:")
        print(list(subset['label']))
        print(fdf)
        bad += 1
        continue
    subset = subset[subset['col'].isin(subset_idxs)]
    fdf = fdf[fdf.columns[fdf_idxs]]
    if len(fdf) > LIMIT_ROWS:
        ids = rand.choice(len(fdf), LIMIT_ROWS, replace=False)
        fdf = fdf.loc[ids]
    truncated = False
    while(True):
        table_str = fdf.to_csv(index=False, header=False, sep='\t')
        ans_str =', '.join(list(subset['label']))
        prompt = f'{table_str}\n\n###\n\n'
        try:
            response = parse_response(get_response(prompt, MODELS[MODEL]))
            break
        except RetryError:
            if not truncated:
                print("Truncating...")
                for col in fdf:
                    try:
                        fdf[col] = fdf[col].str[:1000]
                    except AttributeError:
                        pass
                truncated = True
            else:
                if len(fdf) == 1:
                    raise
                print(f"Reducing {len(fdf)} rows by 1...")
                fdf = fdf.iloc[:-1]

    if len(response) == len(subset):
        df.loc[subset.index, outcol] = response
        good += 1
    else:
        bad_resp += 1
        print(subset['label'])
        print(response)
        import pdb; pdb.set_trace()
        df.loc[subset.index, outcol] = response
        continue
    files.append(subset.iloc[0]['file'])
    cols.append(subset_idxs)

df.to_csv(INFILE, index=False)
