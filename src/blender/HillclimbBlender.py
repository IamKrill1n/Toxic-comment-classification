from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

path = 'kaggle/input/'
comp = 'jigsaw-toxic-comment-classification-challenge/'
TRAIN_DATA_FILE=f'{path}{comp}train.csv.zip'
TEST_DATA_FILE=f'{path}{comp}test.csv.zip'
SAMPLE_SUBMISSION=f'{path}{comp}sample_submission.csv.zip'
SUBMISSION_FOLDER = 'kaggle/working/k-fold/'
OUTPUT = 'kaggle/working/hillclimb-blending/'

from os import chdir, path, getcwd
for i in range(10):
    if path.isfile("checkcwd"):
        break
    chdir(path.pardir)
if path.isfile("checkcwd"):
    pass
else:
    raise Exception("Something went wrong. cwd=" + getcwd())

class HillclimbBlender:
    '''
    Adapt from https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
    and https://www.kaggle.com/code/hhstrand/hillclimb-ensembling - some errors fixed.

    
    Naming convention for files:

    
    `n_sub.csv` for submission files

    `n_oof.csv` for out-of-fold files

    
    where n is id of model:

    1: simple RNN w/ drop-out (Tú)
    2: LSTM w/ drop-out (Tú)
    3: GRU w/ drop-out (Tú)
    4: hybrid LSTM + GRU + feature (T. Thành)
    5: NB with LR (?)
    6: NB with SVM (Vũ, H. Thành)
    7: SVM (Vũ)
    8: NB (H. Thành)
    9: logistic regression (T. Thành)

    '''

    def __init__(
            self,
            train_df_path: str = TRAIN_DATA_FILE,
            test_df_path: str = TEST_DATA_FILE,
            submission_folder_path: str = SUBMISSION_FOLDER,
            sample_submission_path: str = SAMPLE_SUBMISSION,
            output: str = OUTPUT,
            model_num: list = [1, 2, 3, 4]
            ) -> None:
        
        self.train = pd.read_csv(train_df_path).fillna(' ')
        self.test = pd.read_csv(test_df_path).fillna(' ')
        self.labels = self.train.columns[2:]
        self.model_num = model_num
        self.submission_folder = submission_folder_path
        self.output = output

        self.sample_submission = pd.read_csv(sample_submission_path)

        self.best_ensemble = {}
        for label in self.labels:
            self.best_ensemble[label] = list()
        self.best_score = {}
        for label in self.labels:
            self.best_score[label] = 0

        self.oof_files = [pd.read_csv(submission_folder_path + str(num) + '_oof.csv') for num in self.model_num]
        self.sub_files = [pd.read_csv(submission_folder_path + str(num) + '_sub.csv') for num in self.model_num]

    def score_ensemble(self, ensemble: 'list[int]', label: str) -> float:
        '''Calculates and returns score for a particular emsemble.
        '''
        blend_preds = np.zeros(len(self.train))
        
        for model in ensemble:
            blend_preds += self.oof_files[model][label]
            
        blend_preds = blend_preds / len(ensemble)
        score = roc_auc_score(self.train[label], blend_preds)
        
        return score
    
    def find_best_improvement(self, ensemble: 'list[int]', label: str) -> 'tuple[list[int], float]':
        '''Finds the best model to add another 'copy' to the ensemble.
        Returns the modified ensemble and its score.
        '''
        best_score = 0
        best_ensemble = []
        
        for i in range(len(self.oof_files)):
            ensemble = ensemble + [i]
            score = self.score_ensemble(ensemble, label)
            
            if score > best_score:
                best_score  = score
                best_ensemble = ensemble
                
            ensemble = ensemble[:-1]
        
        return best_ensemble, best_score

    def hill_climb(self) -> None:
        '''Finds next best models to add them to each label.
        '''
        for label in self.labels:
            self.best_ensemble[label], self.best_score[label] = self.find_best_improvement(self.best_ensemble[label], label)

    def get_optimal_weights(self) -> 'dict[str, dict[int, float]]':
        '''Calculates and returns the weights from best_ensemble dict.
        '''
        weights = {}
        for label in self.labels:
            weights[label] = {}
            for num in set(self.best_ensemble[label]):
                weights[label][num] = self.best_ensemble[label].count(num) / len(self.best_ensemble[label])
        return weights
    
    def get_optimal_blend(self, optimal_weights: 'dict[str, dict[int, float]]') -> 'pd.DataFrame':
        '''Gets optimal blend from submissions and return a DataFrame.
        '''
        blend = self.sample_submission.copy()
        for label in self.labels:
            blend[label] = 0
            for key in optimal_weights[label]:
                blend[label] += optimal_weights[label][key] * self.sub_files[key][label]
        return blend
    
    def run(self, iter: int = 50) -> None:
        '''Starts hill climbing and writes to file.
        '''
        for i in range(iter):
            print('-------------')
            print('Step', i + 1)    
            self.hill_climb()
            print('Best ensemble:')
            print(self.best_ensemble)
            print('Best score:')
            print(self.best_score)
            print('Avg. AUC score:', np.mean([self.best_score[label] for label in self.labels]))
        print('-------------')

        opt_w = self.get_optimal_weights()
        print('Optimal weights:')
        print(opt_w)
        
        blend = self.get_optimal_blend(opt_w)
        
        blend.to_csv(self.output + 'hillclimb_' + '_'.join([str(i) for i in self.model_num]) + '.csv', index=False)



def main() -> None:

    blender = HillclimbBlender(model_num=[1, 2, 3, 4, 5, 9])
    blender.run(iter=100)
    print()

if __name__ == '__main__':
    main()