import pandas as pd
from sklearn.utils import shuffle
from sklearn.feature_selection import f_classif, chi2

def feature_mask(type):
    df = pd.read_csv("dataset/sound_features.csv", header=None)
    df = df.values
    x = df[:,0:-1]
    y = df[:,-1].astype("float")

    step = 0
    masks = {}
    for vowel in ['A', 'E', 'I', 'O', 'U']:
            train_data_x = x[:, step:step + 26+39]
            step += 27+39

            if type == "ANOVA":
                f_score, p_value = f_classif(train_data_x, y)
            elif type == "chi":
                f_score, p_value = chi2(train_data_x, y)
            mask = p_value <= 0.1
            masks[vowel] = mask

    return masks

mask_cc = feature_mask(type="ANOVA")


def summed_feature_mask():
    df = pd.read_csv("dataset/sound_features.csv", header=None)
    df = df.values
    x = df[:, 0:-1]
    y = df[:, -1].astype("float")

    p_vals =[]
    step = 0
    for _ in ['A', 'E', 'I', 'O', 'U']:
            train_data_x = x[:, step:step + 26]
            step += 27

            f_score, f_p_value = f_classif(train_data_x, y)
            p_vals.append(f_p_value)

    masks = (p_vals[0]<=0.05) | (p_vals[1]<=0.05)| (p_vals[2]<=0.05)| (p_vals[3]<=0.05)| (p_vals[4]<=0.05)
    return masks
