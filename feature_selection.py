from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif

df = pd.read_csv("Dataset/sound_features.csv", header=None)
df = df.values
df = shuffle(df)
x = df[:,0:-1]
y = df[:,-1].astype(float)

step = 0
p_vals =[]
for vowel in ['A', 'E', 'I', 'O', 'U']:
        train_data_x = x[:, step:step + 26]
        scaler = StandardScaler()
        sc_train_data_x = scaler.fit_transform(x[:,step:step+26])
        step += 27

        '''svm = LinearSVC()
        rfe = RFE(svm, n_features_to_select=0.6)
        rfe = rfe.fit(train_data_x, y)
        print(rfe.ranking_)'''

        f_score, f_p_value = f_classif(train_data_x, y)
        p_vals.append(f_p_value)


(p_vals[0]<=0.05) | (p_vals[1]<=0.05)| (p_vals[2]<=0.05)| (p_vals[3]<=0.05)| (p_vals[4]<=0.05)

(p_vals[0]>0.05) & (p_vals[1]>0.05)& (p_vals[2]>0.05)& (p_vals[3]>0.05)& (p_vals[4]>0.05)


def feature_mask():
    mask = [False, False,  True, False,  True, False, False, False,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True]
    return mask