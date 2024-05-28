import multiprocessing as mp
import pandas as pd

from data_preprocessing.get_tools import clean_text_light

def clean_data_multiprocessing(df: 'pd.DataFrame') -> None:
    with mp.Pool(mp.cpu_count()) as pool:
        df['cleaned_data'] = pool.map(clean_text_light, df['comment_text'])