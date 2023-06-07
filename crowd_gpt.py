import pandas as pd
import pickle
import openai
import argparse
from dotenv import load_dotenv
import os
from pathlib import Path
import re
from crowdkit.aggregation import MajorityVote, Wawa, DawidSkene
from googletrans import Translator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff
from prompt_generator import TEMPLATES

import sklearn

STORIES = ['plain', 'game', 'analyst', 'lost']

'''
Purpose: Run multiple prompts with multiple temperatures,
and figure out which configuration is best for entity resolution. We also have functions
to analyze the results and figure out which configuration will give
the best performance.
    
'''

def build_chat(prompt_tmp, example, shot_examples=[]):
    if shot_examples:
        raise ValueError("Few-shot not supported")
    example = example.iloc[0:10]
    for col in example.columns:
        try:
            example[col] = example[col].str[0:256]
        except AttributeError:
            pass

    chat = [{"role": "system", "content": "You are a laconic assistant. You reply with brief, to-the-point answers with no elaboration."}]
    chat += [{"role": "user", "content": prompt_tmp.get_prompt(example.to_csv(index=False, header=False))}]
    return chat

@retry(retry=retry_if_exception_type(openai.error.OpenAIError), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def raw_response(chat, temp_val, lang='english', timeout=30):
    print("Sending: {}".format(chat))
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat,
        temperature=temp_val,
        max_tokens=30
    )
    chat_response = response["choices"][0]["message"]["content"]
    
    if lang != 'english':
        translator = Translator()
        chat_response = translator.translate(chat_response, src=prompt_tmp.lang, dest='english').text
    
    return chat_response

def response_suffwtemp(prompt_tmp, row1, row2, temp_val, timeout=30):
    chat = build_chat(prompt_tmp, row1, row2)
    return raw_response(chat, temp_val, lang=prompt_tmp.lang, timeout=timeout)


def parse_enresponse(response):
    response = response.lower()
    out_lst = []
    label_lst = ['city', 'state', 'category', 'class', 'name', 'description', 
                 'type', 'address', 'age', 'club', 'day', 'location', 'status', 
                 'result', 'year', 'album', 'code', 'position', 'company', 'symbol', 
                 'rank', 'country', 'team', 'county', 'weight', 'language', 'origin', 
                 'genre', 'gender', 'artist', 'collection', 'ranking', 'notes', 
                 'isbn', 'format', 'owner', 'nationality', 'order', 'area', 
                 'publisher', 'region', 'family', 'sex', 'component', 'elevation', 
                 'capacity', 'range', 'duration', 'affiliation', 'teamName', 'plays', 
                 'director', 'credit', 'brand', 'jockey', 'product', 'grades', 
                 'requirement', 'command', 'education', 'birthPlace', 'currency', 
                 'continent', 'depth', 'service', 'industry', 'birthDate', 'sales', 
                 'creator', 'person', 'operator', 'species', 'classification', 
                 'manufacturer', 'fileSize', 'affiliate']
    
    for label in label_lst:
        start = 0
        while True:
            try:
                st_ind = response.index(label, start)
                start = st_ind + 1
                if label == 'age' and st_ind >= 5 and response[(st_ind - 5):].startswith('language'):
                    continue
                out_lst.append((st_ind, label))
            except ValueError:
                break
    
    if out_lst == []:
        return []
    else:
        # sort the list on index, and return the labels
        out_lst = sorted(out_lst, key=lambda x: x[0])
        out_labels = [o[1] for o in out_lst]
        return out_labels

def explode_df(df):
    vec_col(df, 'Ground Truth')
    vec_col(df, 'Story Answer')
    df['Col No'] = df['Ground Truth'].str.len().apply(lambda cols: range(cols))
    mismatched = mismatched_length(df)
    good = df[~mismatched].explode(['Col No', 'Story Answer', 'Ground Truth'], ignore_index=True)
    bad = df[mismatched].explode(['Col No', 'Ground Truth'], ignore_index=True)
    bad['Story Answer'] = "N/A"
    return pd.concat([good, bad])

def vec_col(df, col):
    df[col] = df[col].str.replace("'", "")
    df[col] = df[col].str[1:-1]
    df[col] = df[col].str.split(', ')

def mismatched_length(df):
    return (df['Story Answer'].str[0] == '') | (df['Story Answer'].str.len() != df['Ground Truth'].str.len())

def storysuff(df, story_name, samp_range : list, samp_type, rows, match_prefix, num_reps=10, shots=0, shot_df=None, uniform_shot_offset=None, outdir='matchwsuff', retry=False):
    '''
    Query chatGPT with the given story on the given rows of the match file at different temperatures,
    repeating each prompt a specified number of times at each temperature.

    Parameters
    ----------
    df: pandas.DataFrame
        Source data
    story_name : str
        Story name to use.
    samp_range : list
        List of temperature values to use.
    samp_type : str
        Type of sampling (we could use nucleus sampling too, but we are opting for temperature)
    rows : list
        list of rows to use from the match file, for sampling purposes.
    match_prefix : str
        The prefix describing the match file to be used on output files of this function.
    num_reps : int, optional
        Number of times each prompt should be repeated. The default is 10.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    #story_fname = 'all_prompts/' + story_name + 'prompt_english.pkl'
    #with open(story_fname, 'rb') as fh:
    #    story_tmp = pickle.load(fh)
    story_tmp = TEMPLATES[story_name]
    
    outdir = Path(outdir)

    #shot_yes = shot_df[shot_df['match']].sample(n=len(df)*shots, replace=True, random_state=100).reset_index(drop=True)
    #shot_no = shot_df[~shot_df['match']].sample(n=len(df)*shots, replace=True, random_state=100).reset_index(drop=True)
    shot_yes = None
    shot_no = None

    
    for r_ind in rows:
        dfrow = df.loc[r_ind]
        row_gt = dfrow['labels']
        example_df = pd.read_csv(f'table_data/sato_tables/multionly/K{dfrow["k"]}/{dfrow["fname"]}')
        if len(example_df.columns) != dfrow['num_cols']:
            continue
        for i in range(num_reps):
            for sval in samp_range:
                outname = outdir / ('-'.join([match_prefix, story_name, str(r_ind), f'rep{i}', f'{samp_type}{str(sval).replace(".", "_")}', f'{shots}shot']) + '.csv')
                if outname.exists():
                    if retry:
                        df_orig = pd.read_csv(outname)
                        vec_col(df_orig, 'Ground Truth')
                        vec_col(df_orig, 'Story Answer')
                        if any(mismatched_length(df_orig)):
                            print(f"Retrying {outname}...")
                        else:
                            print(f"{outname} exists...")
                            continue
                    else:
                        print(f"{outname} exists...")
                        continue

                if samp_type == 'temperature':
                    examples = []
                    for i in range(shots):
                        if uniform_shot_offset is not None:
                            shot_idx = uniform_shot_offset * shots + i
                        else:
                            shot_idx = r_ind * shots + i
                        examples.append((shot_yes.loc[shot_idx, 'left'], shot_yes.loc[shot_idx, 'right'], 'YES.'))
                        examples.append((shot_no.loc[shot_idx, 'left'], shot_no.loc[shot_idx, 'right'], 'NO.'))
                    chat = build_chat(story_tmp, example_df, examples)
                    story_response = raw_response(chat, sval)
                    story_answer = parse_enresponse(story_response)
                    if len(story_answer) != dfrow['num_cols']:
                        chat.append({'role': 'assistant', 'content': story_response})
                        chat.append({'role': 'user', 'content': f"Please answer just by listing your best guess for each of the {dfrow['num_cols']} columns"})
                        story_response = raw_response(chat, sval)
                        story_answer = parse_enresponse(story_response)
                else:
                    raise Exception("Sampling Type not supported: {}".format(samp_type))
                
                outdct = {}
                outdct['Row No'] = [r_ind]
                outdct['Rep No'] = [i]
                outdct['Table'] = dfrow['fname']
                outdct['Sampling Type'] = [samp_type]
                outdct['Sampling Param'] = [sval]
                outdct['Shots'] = [shots]
                outdct['Story Name'] = [story_name]
                outdct['Story Response'] = [story_response]
                outdct['Story Answer'] = [story_answer]
                outdct['Ground Truth'] = [row_gt]
                outdf = pd.DataFrame(outdct)
                outdf.to_csv(outname)
    

def crowd_gather(raw_df, temp, outdir):
    '''
    Run different crowd methods on all chatgpt responses.

    Parameters
    ----------
    fullfname : TYPE
        DESCRIPTION.
    temp : TYPE
        DESCRIPTION.
    ditto_dct : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    df = raw_df[raw_df['Sampling Param'] == temp]
    out_schema = ['worker', 'task', 'label']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    raw_labels = df['Story Answer'].tolist()
    new_labels = [max(x, 0) for x in raw_labels] #change -1s to 0s.
    
    out_dct['worker'] = df['Story Name'].tolist()
    out_dct['label'] = new_labels
    matchfiles = df['Match File'].tolist()
    rownos = df['Row No'].tolist()
    tasklst = []
    pairslst = []
    
    for i in range(len(matchfiles)):
        new_el = matchfiles[i] + ':' + str(rownos[i])
        tasklst.append(new_el)
        pairslst.append((matchfiles[i], rownos[i]))
    
    pairsset = set(pairslst)
    
    out_dct['task'] = tasklst
    
    out_df = pd.DataFrame(out_dct)
    
    agg_mv = MajorityVote().fit_predict(out_df)
    agg_wawa = Wawa().fit_predict(out_df)
    agg_ds = DawidSkene(n_iter=10).fit_predict(out_df)
    
    mv_dct = agg_mv.to_dict()
    wawa_dct = agg_wawa.to_dict()
    ds_dct = agg_ds.to_dict()
    
    #res_schema = ['Match File', 'Row No', 'Vote', 'Ditto Answer', 'Ground Truth']
    res_schema = ['Match File', 'Row No', 'Vote', 'Ground Truth']
    mv_res = {}
    wawa_res = {}
    ds_res = {}
    
    for rs in res_schema:
        mv_res[rs] = []
        wawa_res[rs] = []
        ds_res[rs] = []
    
    for pair in pairsset:
        mv_res['Match File'].append(pair[0])
        mv_res['Row No'].append(pair[1])
        wawa_res['Match File'].append(pair[0])
        wawa_res['Row No'].append(pair[1])
        ds_res['Match File'].append(pair[0])
        ds_res['Row No'].append(pair[1])
        
        task_ind = pair[0] + ':' + str(pair[1])
        mv_res['Vote'].append(mv_dct[task_ind])
        wawa_res['Vote'].append(wawa_dct[task_ind])
        ds_res['Vote'].append(ds_dct[task_ind])
        
        pair_df = df[(df['Match File'] == pair[0]) & (df['Row No'] == pair[1])]
        pair_gt = pair_df['Ground Truth'].unique().tolist()[0]
        mv_res['Ground Truth'].append(pair_gt)
        wawa_res['Ground Truth'].append(pair_gt)
        ds_res['Ground Truth'].append(pair_gt)
        
        #mv_res['Ditto Answer'].append(ditto_dct[pair[1]])
        #wawa_res['Ditto Answer'].append(ditto_dct[pair[1]])
        #ds_res['Ditto Answer'].append(ditto_dct[pair[1]])
    
    mv_df = pd.DataFrame(mv_res)
    wawa_df = pd.DataFrame(wawa_res)
    ds_df = pd.DataFrame(ds_res)
    
    mv_df.to_csv(outdir + '/MajorityVote_results' + '-temperature' + str(temp).replace('.', '_') + '.csv', index=False)
    wawa_df.to_csv(outdir + '/Wawa_results' + '-temperature' + str(temp).replace('.', '_') + '.csv', index=False)
    ds_df.to_csv(outdir + '/DawidSkene_results' + '-temperature' + str(temp).replace('.', '_') + '.csv', index=False)

def get_stats_simple(df):
    macro_f1 = sklearn.metrics.f1_score(df['Ground Truth'], df['Story Answer'], labels=df['Ground Truth'].unique(), average='macro')
    weighted_f1 = sklearn.metrics.f1_score(df['Ground Truth'], df['Story Answer'], labels=df['Ground Truth'].unique(), average='weighted')
    return (macro_f1, weighted_f1)

        

def get_stats(method_names, temperatures, story_names, outdir):
    '''
    Generate a csv to compare the precision, recall, f1 of crowd methods, individual prompt responses, etc. to baselines

    Parameters
    ----------
    method_names : TYPE
        DESCRIPTION.
    temps : TYPE
        DESCRIPTION.
    story_names : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    method2df = {}
    story2df = {}
    for mn in method_names:
        for temp in temperatures:
            strtemp = str(temp).replace('.', '_')
            method2df[(mn, temp)] = pd.read_csv(f'{outdir}/{mn}_results-temperature{strtemp}.csv')

    for temp in temperatures:
        strtemp = str(temp).replace('.', '_')
        story_df = pd.read_csv(f'{outdir}/per_promptresults_tmperature{strtemp}.csv')
        for sn in story_names:
            story2df[(sn, temp)] = story_df[story_df['Prompt'] == sn]
    truth_df = method2df[(method_names[0], temperatures[0])][['Match File', 'Row No', 'Ground Truth']].copy()

    out_dict = {'Dataset': [], 'Method': [], 'Temp': [], 'Crowd': [], 'F1': [], 'Precision': [], 'Recall': []}

    for mn in method_names:
        for temp in temperatures:
            full_df = method2df[(mn, temp)]
            full_df['Vote'] = (full_df['Vote'] == 1)

            for dataset, df in full_df.groupby('Match File'):
                our_tps = df[(df['Vote'] == df['Ground Truth']) & (df['Ground Truth'] == True)].shape[0]
                our_tns = df[(df['Vote'] == df['Ground Truth']) & (df['Ground Truth'] == False)].shape[0]
                our_fps = df[(df['Vote'] != df['Ground Truth']) & (df['Ground Truth'] == False)].shape[0]
                our_fns = df[(df['Vote'] != df['Ground Truth']) & (df['Ground Truth'] == True)].shape[0]
            
                our_precision = our_tps / (our_tps + our_fps)
                our_recall = our_tps / (our_tps + our_fns)
                our_f1 = 2 * (our_precision * our_recall) / (our_precision + our_recall)

                out_dict['Dataset'].append(dataset.split('/')[1].split('.')[0])
                out_dict['Method'].append(mn)
                out_dict['Temp'].append(temp)
                out_dict['Crowd'].append(True)
                out_dict['F1'].append(our_f1)
                out_dict['Precision'].append(our_precision)
                out_dict['Recall'].append(our_recall)
            
    for sn in story_names:
        for temp in temperatures:
            full_df = story2df[(sn, temp)]
            full_df['Vote'] = full_df['Majority']
            full_df = full_df.merge(truth_df, on=['Match File', 'Row No'])

            for dataset, df in full_df.groupby('Match File'):
                our_tps = df[(df['Vote'] == df['Ground Truth']) & (df['Ground Truth'] == True)].shape[0]
                our_tns = df[(df['Vote'] == df['Ground Truth']) & (df['Ground Truth'] == False)].shape[0]
                our_fps = df[(df['Vote'] != df['Ground Truth']) & (df['Ground Truth'] == False)].shape[0]
                our_fns = df[(df['Vote'] != df['Ground Truth']) & (df['Ground Truth'] == True)].shape[0]
            
                our_precision = our_tps / (our_tps + our_fps)
                our_recall = our_tps / (our_tps + our_fns)
                our_f1 = 2 * (our_precision * our_recall) / (our_precision + our_recall)

                out_dict['Dataset'].append(dataset.split('/')[1].split('.')[0])
                out_dict['Method'].append(sn)
                out_dict['Temp'].append(temp)
                out_dict['Crowd'].append(False)
                out_dict['F1'].append(our_f1)
                out_dict['Precision'].append(our_precision)
                out_dict['Recall'].append(our_recall)

    pd.DataFrame(out_dict).to_csv(f'{outdir}/stats.csv', index=False)


def query(args):
    load_dotenv()
    openai.api_key = os.getenv(f"OPENAI_API_KEY{args.key}")
    uniform_shot_offset = None
    if args.uniform_shots:
        uniform_shot_offset = args.uniform_shot_offset
    for num_reps in range(1, args.reps + 1):
        print(f"Rep {num_reps}:")
        for d in args.datasets:
            print(f"Dataset {d}:")
            maindf = pd.read_csv(d)
            traindf = None
            match_prefix = d.split('.')[0]
            rep_row = maindf['num_cols'].to_dict().keys()
            for s in args.stories:
                print(f"Story {s}")
                storysuff(maindf, s, args.temps, 'temperature', rep_row, match_prefix, num_reps=num_reps, shots=args.shots, shot_df=traindf, uniform_shot_offset=uniform_shot_offset, outdir=args.rawdir, retry=args.retry)


def combine(args):
    os.makedirs(args.outdir, exist_ok=True)
    #regex = re.compile(r'(?P<dataset>[A-Za-z]+(-[A-Za-z]+)?)-(?P<story>[a-z]+)-(?P<row>[0-9]+)-rep(?P<rep>[0-9]+)-temperature(?P<temp>[0-2])_0-(?P<shot>[0-9])shot.csv')
    small_dfs = []
    big_dfs = []
    counter = 0
    print("Fixing '^M' characters...")
    os.system(f"bash ./fix_newline.sh {args.rawdir}")
    for f in Path(args.rawdir).glob(f'*.csv'):
        df = pd.read_csv(f)
        small_dfs.append(df)
        if len(df) > 1:
            import pdb; pdb.set_trace()
        counter += 1
        if counter % 1000 == 0:
            print(f'{counter}...')
            big_dfs.append(pd.concat(small_dfs))
            small_dfs = []
    print("Concatting...")
    df = pd.concat(big_dfs)
    print("Writing...")
    df.to_csv(f'{args.outdir}/full.csv', index=False)

def analyze(args):
    result_file = f"{args.outdir}/full.csv"
    df = pd.read_csv(result_file)
    df = df[~df['Sampling Param'].isna()]
    df['Rep No'] = df['Rep No'].astype(float).astype(int)
    df['Row No'] = df['Row No'].astype(float).astype(int)
    if args.reps is not None:
        df = df[df['Rep No'] <= args.reps]
    df = explode_df(df)
    '''
    for temp in args.temps:
        print(f"Temp {temp}:")
        crowd_gather(df, temp, args.outdir)
        perprompt_majorities(df, temp, args.outdir)
    get_stats(["MajorityVote", "Wawa", "DawidSkene"], args.temps, args.stories, args.outdir)
    '''
    for story in args.stories:
        df_story = df[df['Story Name'] == story]
        macro_f1, weighted_f1 = get_stats_simple(df_story)
        import pdb; pdb.set_trace()

def retry(args):
    load_dotenv()
    openai.api_key = os.getenv(f"OPENAI_API_KEY{args.key}")
    result_file = f"{args.outdir}/full.csv"
    mf2df = {}
    full_df = pd.read_csv(result_file)
    to_fix = full_df[full_df['Story Answer'] == -1.0]
    total = len(to_fix)
    fixed = 0
    progress = 0
    for idx, row in to_fix.iterrows():
        mf = row['Match File']
        if mf not in mf2df:
            mf2df[mf] = pd.read_csv(mf)
        df = mf2df[mf]
        source_row = mf2df[mf].loc[row['Row No']]
        story_tmp = TEMPLATES[row['Story Name']]
        chat = build_chat(story_tmp, source_row['left'], source_row['right'])
        response = row['Story Response']
        if type(response) == str:
            chat.append({'role': 'assistant', 'content': row['Story Response']})
        chat.append({'role': 'user', 'content': "Please answer just YES or NO. If you are uncertain, make your best guess."})
        resp = raw_response(chat, row['Sampling Param'])
        answer = parse_enresponse(resp)
        progress += 1
        if answer != -1:
            fixed += 1
        print(f"Fixed {fixed}/{progress}; total {total}")
        full_df.loc[idx, 'Story Answer'] = answer
        if progress % 100 == 0:
            full_df.to_csv(result_file, index=False)
    full_df.to_csv(result_file, index=False) 
        
    

if __name__=='__main__':
    #this is a representative row, 
    #in that the first is a true positive, next is false positive, next is true negative, next is false negative
    # rep_row = [23, 25, 46, 0, 29, 31]
    # rep_row = [2, 12, 30, 0, 1, 3, 46, 201, 302, 44, 136, 207]
    # ditto_dct = {2 : True, 12 : True, 30 : True, 0 : False, 1 : False, 3 : False, 46 : True, 201 : True, 302 : True, 44 : False, 136 : False, 207 : False}
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    parser_query = subparsers.add_parser('query')
    parser_query.add_argument("--stories", nargs='+', default=STORIES, choices=list(TEMPLATES.keys()))
    parser_query.add_argument("--datasets", nargs='+', default=['full_testset.csv'])
    parser_query.add_argument("--reps", type=int, default=10)
    parser_query.add_argument("--temps", type=float, nargs='+', default=[2.0])
    parser_query.add_argument("--key", type=int, required=True)
    parser_query.add_argument("--shots", type=int, default=0)
    parser_query.add_argument("--uniform-shots", action='store_true')
    parser_query.add_argument("--uniform-shot-offset", type=int, default=0)
    parser_query.add_argument("--rawdir", required=True)
    parser_query.add_argument("--retry", action="store_true")
    parser_query.set_defaults(func=query)

    parser_combine = subparsers.add_parser('combine')
    parser_combine.add_argument("--rawdir", required=True)
    parser_combine.add_argument("--outdir", required=True)
    parser_combine.set_defaults(func=combine)

    parser_analyze = subparsers.add_parser('analyze')
    parser_analyze.add_argument("--reps", type=int, default=10)
    parser_analyze.add_argument("--temps", type=float, nargs='+', default=[2.0])
    parser_analyze.add_argument("--stories", nargs='+', default=STORIES, choices=list(TEMPLATES.keys()))
    parser_analyze.add_argument("--outdir", required=True)
    parser_analyze.set_defaults(func=analyze)

    parser_retry = subparsers.add_parser('retry')
    parser_retry.add_argument("--key", type=int, required=True)
    parser_retry.add_argument("--rawdir", required=True)
    parser_retry.set_defaults(func=retry)

    args = parser.parse_args()
    args.func(args)
