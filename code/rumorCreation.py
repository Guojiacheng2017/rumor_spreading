# import random as rd
# print(rd.randint(0, 1))
# def generateRumor():

import numpy as np
import pandas as pd
from random import choice
# x = np.random.normal(size=(2, 3))
user_count = 0
rumorSize = 100

# >>> rumor generated indicating by labels -1 means rumors and 1 means true (no misinformation)
rumorLabel = np.random.rand(rumorSize)
for idx,rumor in enumerate(rumorLabel):
    if rumor >= 0.5:
        rumorLabel[idx] = 1
    else:
        rumorLabel[idx] = -1

# print(rumorLabel)

rumorList = pd.DataFrame(rumorLabel, columns=['labels'])
# print(rumorList)


def userDecision(rumorSize, consensus_mechanism):
    if consensus_mechanism == 1:
        userBehavior = abs(np.random.normal(size=rumorSize, loc = np.random.rand(), scale = 0.6))

        for idx, ub in enumerate(userBehavior):
            # print(idx)
            if ub >= 0.7:
                userBehavior[idx] = rumorLabel[idx]
            else:
                userBehavior[idx] = round(np.random.rand() * 2 - 1)
                # some information is right, some is wrong and some is not this man's business
    else:
        userBehavior = abs(np.random.normal(size=rumorSize, loc = np.random.rand(), scale = 0.6))
        for idx, ub in enumerate(userBehavior):
            # print(idx)
            if ub >= 0.5:
                userBehavior[idx] = 1
            else:
                userBehavior[idx] = -1
    rumorList['user_' + str(user_count)] = userBehavior
    
    # print(userBehavior)
    # print(sum(rumorList * userBehavior))


def sigmoid(x):
    rst = 1/(1 + np.exp(-x))
    # print(rst)
    if rst >= 0.5:
        rst = 1
    else:
        rst = -1
    return rst



for i in range(10):
    userDecision(rumorSize, 0)
    user_count += 1

# print(rumorList)


# random select 6 element in the row for the voting
# 1. separate the dataframe
userForm = rumorList.iloc[:,1:user_count+1]
# print(userForm)
# 2. select random element from every row
predList = []


# 3. judge whether it is a rumor by voting (Proof-of-Stake consensus mechanism)
cm = 0
if cm == 1:
    userForm['prediction'] = userForm.loc[0: rumorSize].sum(axis=1)
    userForm['prediction'] = userForm['prediction'].apply(lambda x: sigmoid(x))
else:
    userForm['prediction'] = userForm['user_0']
# print(userForm)


# ----- Evaluation -----
pred = userForm['prediction'].to_numpy()
labels = rumorList['labels'].to_numpy()

from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(labels, pred)
precision = precision_score(labels, pred)
recall = recall_score(labels, pred)
f1 = f1_score(labels, pred)

print(cm, precision, recall, f1)
f = sns.heatmap(cm, annot=True, fmt='d')
plt.savefig("../rst/cm0.png")


# import matplotlib.pyplot as plt
# from sklearn.metrics import ConfusionMatrixDisplay
# disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=labels)

# # NOTE: Fill all variables here with default values of the plot_confusion_matrix
# disp = disp.plot(include_values=confusion, cmap=plt.cm.Blues)

# plt.show()

print(rumorList)
print(userForm)

# print(pred)
# print(labels)