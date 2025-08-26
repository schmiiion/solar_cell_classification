import torch as t
import torch.nn as nn
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from model import DenseNetMultiLabel
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
df = pd.read_csv('data.csv')
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_ds = ChallengeDataset(train_df, mode='train')
val_ds = ChallengeDataset(val_df, mode='val')

train_dl = t.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = t.utils.data.DataLoader(val_ds, batch_size=64, shuffle=True)

# create an instance of our ResNet model
model = DenseNetMultiLabel()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
criterion = nn.BCELoss()
optimizer = t.optim.Adam(model.parameters(), lr=0.001)

# go, go, go... call fit on trainer
trainer = Trainer(model, criterion, optimizer, train_dl, val_dl, cuda=False)
res = trainer.fit(epochs=2)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')