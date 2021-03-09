from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from keras.metrics import BinaryAccuracy
from keras.optimizers import Adam
import pandas as pd
import numpy as np
import cv2
import math
import statistics
import neural_nets
import feature_selection

def calc_label(true_val, pred_val):
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
    return TP, TN, FP, FN

def safe_div(x,y):
    if y == 0:
        return 0
    return x / y

def calc_metrics(true_val, pred_val):
    TP, TN, FP, FN = calc_label(true_val=true_val, pred_val=pred_val)
    acc = safe_div(TP+TN, TP+TN+FP+FN)
    sens = safe_div(TP, TP + FN)
    spec = safe_div(TN, TN + FP)
    mcc =  safe_div((TP * TN) - (FP * FN), math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    return acc, sens, spec, mcc


df = pd.read_csv("Dataset/sound_features.csv", header=None)
df = df.values
df = shuffle(df)
x = df[:,0:-1]
y = df[:,-1].astype(float)


groups = [i for i in range(50) for _ in range(2)]
groups = [i for i in range(2) for _ in range(50)]
logo = LeaveOneGroupOut()
feature_mask = feature_selection.feature_mask()

accuracy, sensitivity, specificity, MCC = [],[],[],[]
for train_id, test_id in logo.split(x, y, groups=groups):
    train_data_y = y[train_id]
    test_data_y = y[test_id]

    step = 0
    decision_1 = 0
    decision_2 = 0
    for vowel in ['A', 'E', 'I', 'O', 'U']:
        scaler = StandardScaler()
        train_data_x = scaler.fit_transform(x[train_id, step:step+26])
        test_data_x = scaler.transform(x[test_id,step:step+26])
        train_data_x = train_data_x[:,feature_mask]
        test_data_x = test_data_x[:,feature_mask]

        img_train_path = x[train_id, step+26]
        img_test_path = x[test_id, step+26]
        step += 27

        img_train_data = []
        img_test_data = []
        for path in img_train_path:
            image = cv2.imread(path)
            image = image / 255.0
            img_train_data.append(image)
        for path in img_test_path:
            image = cv2.imread(path)
            image = image / 255.0
            img_test_data.append(image)

        img_train_data = np.array(img_train_data)
        img_test_data = np.array(img_test_data)

        model = neural_nets.create_mixed_model()
        model.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-4), metrics=[BinaryAccuracy()])
        print('Vowel "{}" model training started'.format(vowel))
        model.fit(
            x=[train_data_x, img_train_data], y=train_data_y,
            validation_data=([test_data_x, img_test_data], test_data_y),
            epochs=1, batch_size=32, shuffle=True)

        predictions = model.predict_on_batch([test_data_x, img_test_data])
        decision_1 += predictions[0]
        decision_2 += predictions[1]

        print('Vowel "{}" model trained'.format(vowel))

    ### All models were trained and predicted
    decision = []
    decision.append(1 if decision_1 > 2.5 else 0)
    decision.append(1 if decision_2 > 2.5 else 0)

    acc, sens, spec, mcc = calc_metrics(true_val=test_data_y, pred_val=decision)
    accuracy.append(acc)
    sensitivity.append(sens)
    specificity.append(spec)
    MCC.append(mcc)

### Aggregation of all CV splits
final_acc = statistics.mean(accuracy)
final_sens = statistics.mean(sensitivity)
final_spec = statistics.mean(specificity)
final_mcc = statistics.mean(MCC)
