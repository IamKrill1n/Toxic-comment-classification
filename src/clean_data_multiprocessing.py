import multiprocessing as mp
import pandas as pd
import numpy as np


from src.get_tools import clean_text_light

def clean_data_multiprocessing(df: 'pd.DataFrame') -> None:
    with mp.Pool(mp.cpu_count()) as pool:
        df['cleaned_data'] = pool.map(clean_text_light, df['comment_text'])
        # fill empty cell
        df['cleaned_data'] = df['cleaned_data'].apply(lambda x: np.nan if isinstance(x, str) and (x.isspace() or x == '') else x)
        df['cleaned_data'] = df['cleaned_data'].fillna('something')