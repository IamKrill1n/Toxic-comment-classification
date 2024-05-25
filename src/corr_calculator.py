import pandas as pd
from scipy.stats import ks_2samp

def corr(first_file, second_file):
    # assuming first column is `class_name_id`
    first_df = pd.read_csv(first_file, index_col=0)
    second_df = pd.read_csv(second_file, index_col=0)
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    for class_name in class_names:
        # all correlations
        # print('\n Class: %s' % class_name)
        pearson_corr = first_df[class_name].corr(second_df[class_name], method='pearson')
        kendall_corr = first_df[class_name].corr(second_df[class_name], method='kendall')
        spearman_corr = first_df[class_name].corr(second_df[class_name], method='spearman')
        ks_stat, p_value = ks_2samp(first_df[class_name].values, second_df[class_name].values)
        # print(' Pearson\'s correlation score: %0.6f' % pearson_corr)
        # print(' Kendall\'s correlation score: %0.6f' % kendall_corr)
        # print(' Spearman\'s correlation score: %0.6f' % spearman_corr)
        # print(' Kolmogorov-Smirnov test:    KS-stat = %.6f    p-value = %.3e\n' % (ks_stat, p_value))

        # Store the values in a variable
        results = {
            'class_name': class_name,
            'pearson_corr': pearson_corr,
            'kendall_corr': kendall_corr,
            'spearman_corr': spearman_corr,
            'ks_stat': ks_stat,
            'p_value': p_value
        }

    # Return the correlation values
    return results
