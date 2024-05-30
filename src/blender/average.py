import os
import pandas as pd

submission_path = 'kaggle/working/blender/'  
submission_files = os.listdir(submission_path)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
ensemble_submission = pd.DataFrame()

for file in submission_files:
    if file.endswith('.csv') and file != 'average_ensemble.csv':
        file_path = os.path.join(submission_path, file)
        submission_data = pd.read_csv(file_path)
        # print(file_path, submission_data.head())
        if ensemble_submission.empty:
            ensemble_submission = submission_data.copy()
            ensemble_submission[label_cols] = ensemble_submission[label_cols] * 0
            continue
        ensemble_submission[label_cols] = ensemble_submission[label_cols].add(submission_data[label_cols])
        ensemble_submission['id'] = submission_data['id']

print(len(submission_files))
ensemble_submission[label_cols] = ensemble_submission[label_cols] / len(submission_files)
ensemble_submission
ensemble_submission.to_csv(submission_path + '/average_ensemble.csv', index=False)