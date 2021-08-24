import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from keras.metrics import BinaryAccuracy
from keras.optimizers import Adam
import pandas as pd
import numpy as np
import cv2
import math
import neural_nets
import feature_selection
import csv
from keras import backend as K
import tensorflow as tf


def calc_label(true_val, pred_val):
    assert len(true_val) == len(pred_val), "The number of actual and estimated values ​​does not match"

    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(pred_val)):
        if true_val[i] == 0:
            if pred_val[i] == 0:
                TN += 1
            else:
                FP += 1
        elif true_val[i] == 1:
            if pred_val[1] == 1:
                TP += 1
            else:
                FN += 1
    return np.array([TP, TN, FP, FN])


def safe_div(x,y):
    if y == 0:
        return 0
    return x / y

def import_images(path_list):
    img_data = []
    for path in path_list:
        image = cv2.imread(path)
        image = image/255.0
        img_data.append(image)
    return np.array(img_data)

def log_csv(text):
    with open('TF_labels.csv', mode='w') as employee_file:
        csvwriter = csv.writer(employee_file)
        csvwriter.writerow(text)


df = pd.read_csv("dataset/sound_features.csv", header=None)
df = df.values
x = df[:,0:-1]
y = df[:,-1].astype(float)

groups = [i for i in range(50) for _ in range(2)]
logo = LeaveOneGroupOut()
feature_mask = feature_selection.feature_mask(type="ANOVA")

TF_labels = np.array([0,0,0,0]) # True-false-positive-negative prediction labels (TP,TN,FP,FN)
subset_id = 0
for train_id, test_id in logo.split(x, y, groups=groups):
    subset_id += 1
    if subset_id < 25:      # Jump to an arbitrary subset
        pass
    else:

        train_data_y = y[train_id]
        test_data_y = y[test_id]

        step = 0
        decisions = np.array([0,0])
        for vowel in ['A', 'E', 'I', 'O', 'U']:     # Train separate neural networks for different vowels
            tb_callback = TensorBoard(log_dir="tensorboard/model_{}{}".format(vowel, subset_id), histogram_freq=0, write_graph=False)
            scaler = StandardScaler()
            train_data_x = scaler.fit_transform(x[train_id, step:step+26+39].astype("float32"))      # Fit z-score normalizer only on training data and apply both on train and test set
            test_data_x = scaler.transform(x[test_id,step:step+26+39].astype("float32"))
            train_data_x = train_data_x[:,feature_mask[vowel]].astype("float32")   # Feature selection with previously defined mask
            test_data_x = test_data_x[:,feature_mask[vowel]].astype("float32")

            img_train_path = x[train_id, step+26+39]
            img_test_path = x[test_id, step+26+39]
            img_train_data = import_images(img_train_path)
            img_test_data = import_images(img_test_path)

            step += 27+39

            model = neural_nets.create_mixed_model(dim=train_data_x.shape[1])
            model.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-4), metrics=[BinaryAccuracy()])
            print('Vowel "{}"{} model training started'.format(vowel, subset_id))
            model.fit(
                x=[train_data_x, img_train_data], y=train_data_y,
                #validation_data=([test_data_x, img_test_data], test_data_y), callbacks=[tb_callback],
                epochs=600, batch_size=16, shuffle=True#,verbose=0
            )

            predictions = model.predict_on_batch([test_data_x, img_test_data])
            decisions = np.add(decisions, np.squeeze(predictions).round(0))
            K.clear_session()
            tf.compat.v1.reset_default_graph()

        ### All models were trained and predicted
        for i in range(len(decisions)):
            decisions[i] = 1 if decisions[i] > 2.5 else 0

        TF_labels = np.add(TF_labels, calc_label(true_val=test_data_y, pred_val=decisions))
        # subset_id += 1
        log_csv(["TP:{},TN:{},FP:{},FN:{}".format(TF_labels[0],TF_labels[1],TF_labels[2],TF_labels[3])])

### Aggregation of all CV splits
TP, TN, FP, FN = TF_labels
accuracy = safe_div(TP+TN, TP+TN+FP+FN)
sensitivity = safe_div(TP, TP + FN)
specificity = safe_div(TN, TN + FP)
MCC =  safe_div((TP * TN) - (FP * FN), math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))

print("____ALL SUBSETS TRAINED____")
print("accuracy: {}, sensitivity: {}, specificity: {}, MCC: {}".format(accuracy, sensitivity, specificity, MCC))
