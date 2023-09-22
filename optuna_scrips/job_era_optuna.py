#!/usr/bin/env python
# coding: utf-8

import sys
import datetime
log_file = open(sys.argv[3], "a")
log_file.write("START LOG\n")
log_file.close()

log_file = open(sys.argv[3], "a")
log_file.write("Time: %s\n" %(str(datetime.datetime.now())))
log_file.close()

log_file = open(sys.argv[3], "a")
log_file.write("Logs for ERA model optuna optimization\n")
log_file.close()




log_file = open(sys.argv[3], "a")
log_file.write("Script name: %s, path: %s, n trials: %s\n" %(sys.argv[0], sys.argv[1], sys.argv[2]))
log_file.close()


log_file = open(sys.argv[3], "a")
log_file.write("___________________________________________________\n")
log_file.close()

log_file = open(sys.argv[3], "a")
log_file.write("\n")
log_file.close()

# In[1]:


import xarray as xr
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

import optuna

import time

log_file = open(sys.argv[3], "a")
log_file.write("Imports complete\n")
log_file.close()

log_file = open(sys.argv[3], "a")
log_file.write("\n")
log_file.close()


name_of_script = sys.argv[0]
path = sys.argv[1]
n_trials = int(sys.argv[2])
log_file = open(sys.argv[3], "a")
log_file.write("Args set\n")
log_file.close()

log_file = open(sys.argv[3], "a")
log_file.write("\n")
log_file.close()


# In[2]:


device = "cuda" if torch.cuda.is_available() else "cpu"
log_file = open(sys.argv[3], "a")
log_file.write("Device: " + str(device)+"\n")
log_file.close()

log_file = open(sys.argv[3], "a")
log_file.write("\n")
log_file.close()


# In[3]:

log_file = open(sys.argv[3], "a")
log_file.write("Loading data:\n")
log_file.close()

xr_train = xr.open_dataset(path+'xr_train_timestamped.netcdf', engine='netcdf4')
xr_test = xr.open_dataset(path+'xr_test_timestamped.netcdf', engine='netcdf4')

log_file = open(sys.argv[3], "a")
log_file.write("- xr data loaded\n")
log_file.close()


df_metadata_train = pd.read_csv(path + 'df_metadata_train.csv', index_col=0)
df_metadata_train.index = df_metadata_train.index.astype(str)

df_metadata_test = pd.read_csv(path + 'df_metadata_test.csv', index_col=0)
df_metadata_test.index = df_metadata_test.index.astype(str)

log_file = open(sys.argv[3], "a")
log_file.write("- metadata loaded\n")
log_file.close()


df_distances_test = pd.read_csv(path+'df_distances_test.csv', index_col=0)
df_distances_train = pd.read_csv(path+'df_distances_train.csv', index_col=0)
df_distances_test.index = df_distances_test.index.astype(str)
df_distances_train.index = df_distances_train.index.astype(str)

log_file = open(sys.argv[3], "a")
log_file.write("- distances loaded\n")
log_file.close()


era = xr.open_dataset(path+'era_stamped.nc')

log_file = open(sys.argv[3], "a")
log_file.write("era data loaded\n")
log_file.close()

log_file = open(sys.argv[3], "a")
log_file.write("All data ready\n")
log_file.close()

log_file = open(sys.argv[3], "a")
log_file.write("\n")
log_file.close()

# In[4]:


from torch.utils.data import Dataset


def get_seq_length(past_dict = {'year': 0, 'month': 0, 'week': 0, 'day': 1, 'hour': 0},
                   future_dict = {'year': 0, 'month': 0, 'week': 0, 'day': 0, 'hour': 1},
                   max_len = 133992):
    look_up = (past_dict['hour'] * 4 + past_dict['day'] * 24 * 4
            + past_dict['month'] *30 * 24 * 4 + past_dict['year'] * 12 * 30 * 24 * 4)
    look_ahead = (future_dict['hour'] * 4 + future_dict['day'] * 24 * 4
                + future_dict['month'] *30 * 24 * 4 + future_dict['year'] * 12 * 30 * 24 * 4)

    if look_up+look_ahead > max_len:        
        log_file = open(sys.argv[3], "a")
        log_file.write("Not enough data to accomodate requested sequence lengths, returning 1 and 1\n")
        log_file.close()

        look_up, look_ahead = 1, 1

    return look_up, look_ahead


class PVDataset_ERA(Dataset):
    def __init__(self, X,
               era, df_metadata, df_distances=None, look_up=1, look_ahead=1, with_neighbours=False,
               cut_off_km=64, n_neighbours=8):

        '''
        X = xarray dataset
        look_up = int, length of sequence of past observations to use for prediction
        look_ahead = int, length of sequence to predict
        era = xarray dataset of era5 data
        '''

        self.X = X
        self.stations = list(X.keys())
        self.look_up = look_up
        self.look_ahead = look_ahead
        self.shape_0 = X.dims['datetime']
        self.shape_1 = len(self.stations)
        self.data_len = self.shape_0 // (self.look_up + self.look_ahead)
        self.df_metadata = df_metadata
        self.era_keys = list(era.keys())
        self.era = era

        self.with_neighbours = with_neighbours
        if with_neighbours:
            self.df_distances = df_distances
            self.cut_off_km = cut_off_km
            self.n_neighbours = n_neighbours

    def __len__(self):
        return self.data_len * self.shape_1

    def __getitem__(self, i):

        def get_neighbours(ss_id):
            distances = self.df_distances[ss_id].sort_values().copy()
            distances = distances[distances > self.cut_off_km].copy()

            if len(distances) < self.n_neighbours:
                return np.random.choice(distances.index.values, self.n_neighbours, replace=True)

            else:
                return distances.index.values[list(range(0, len(distances), len(distances)//self.n_neighbours))[:self.n_neighbours]]


        ss_num = i // self.data_len
        ss_id = self.stations[ss_num]
        i = (i % self.shape_1) * (self.look_up + self.look_ahead)

        t = i + self.look_up

#        while sum(self.X[ss_id].data[t-self.look_up : t]) < 0.5:
#            i = np.random.choice(range(self.__len__()), 1)[0]

#            ss_num = i // self.data_len
#            ss_id = self.stations[ss_num]
#            i = (i % self.shape_1) * (self.look_up + self.look_ahead)

#            t = i + self.look_up

        x_item = np.zeros((self.look_up, 5+len(self.era_keys)))
        y_item = np.zeros((self.look_ahead, 1))

        if self.with_neighbours:
            x_item = np.zeros((self.look_up, 5+len(self.era_keys)+self.n_neighbours))
            neighbours = get_neighbours(ss_id)

        x_item[:, 0], y_item = self.X[ss_id].data[t-self.look_up : t], self.X[ss_id].data[t : t+self.look_ahead]

        era_slice = self.era.sel(datetime=self.X['datetime'].data[t-self.look_up : t], ss_id=ss_id).copy()
  
        for i, key in enumerate(self.era_keys):
            x_item[:, i+1] = era_slice[key].data

        if self.with_neighbours:
            for i, ss in enumerate(neighbours):
                x_item[:, i+1+len(self.era_keys)] = self.X[ss].data[t-self.look_up : t]

        x_item[:, -4] = self.X.date_sin.data[t-self.look_up : t]
        x_item[:, -3] = self.X.date_cos.data[t-self.look_up : t]
        x_item[:, -2] = self.X.time_sin.data[t-self.look_up : t]
        x_item[:, -1] = self.X.time_sin.data[t-self.look_up : t]

        return torch.Tensor(x_item).float(), torch.Tensor(y_item).float()


# In[5]:


from torch.utils.data import DataLoader

def loaders(xr_test, xr_train, era, batch_size, dict_look_up, dict_look_ahead,
            df_metadata_train, df_metadata_test,
            cut_off_km=64, n_neighbours=8,
            df_distances_train=None, df_distances_test=None, with_neighbours=False):

    if with_neighbours and (df_distances_train is None or df_distances_test is None):
        raise ValueError("With_neighbours requested, but no distances passed")
    

    max_len = len(xr_train['datetime'].data)

    look_up, look_ahead = get_seq_length(past_dict=dict_look_up, future_dict=dict_look_ahead, max_len=max_len)

    train_dataset = PVDataset_ERA(X=xr_train, era=era, df_metadata=df_metadata_train, df_distances=df_distances_train, look_up=look_up, look_ahead=look_ahead,
                            cut_off_km=cut_off_km, n_neighbours=n_neighbours, with_neighbours=with_neighbours)
    test_dataset =  PVDataset_ERA(X=xr_test, era=era, df_metadata=df_metadata_test, df_distances=df_distances_test, look_up=look_up, look_ahead=look_ahead,
                            cut_off_km=cut_off_km, n_neighbours=n_neighbours, with_neighbours=with_neighbours)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    log_file = open(sys.argv[3], "a")
    log_file.write("Expected number of steps is %d\n" %((train_dataset.__len__()  + batch_size - 1) // batch_size))
    log_file.close()


    return train_loader, test_loader, look_ahead


# In[6]:


class LSTM_model(nn.Module):

    def __init__(self, hidden_size,
               num_layers,
               out_features,
               input_size,
               dropout=0):

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout
                            )

        self.fc = nn.Linear(in_features=hidden_size, out_features=out_features)

    def forward(self, x):

        batch_size = x.size(0)

        h_t = torch.zeros(self.num_layers,
                          batch_size,
                          self.hidden_size).to(device)
        c_t = torch.zeros(self.num_layers,
                          batch_size,
                          self.hidden_size).to(device)


        _, (h_out, _) = self.lstm(x, (h_t, c_t))

        h_out = h_out[0].view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out


# In[7]:


def train_epoch(model, train_loader, optimizer, loss_function, print_every=500):
    loss_array = []
    model.train(True)
    running_loss = 0.

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)

        loss = loss_function(output, y_batch.squeeze())
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % print_every == 0:
            loss_array.append(running_loss/print_every)

            log_file = open(sys.argv[2], "a")
            log_file.write("Batch %d, loss: %f" %(batch_index+1, running_loss/print_every))
            log_file.close()

            running_loss=0.

    loss_array.append(running_loss/((batch_index+1)%1000))
    
    log_file = open(sys.argv[3], "a")
    log_file.write("\n")
    log_file.close()

    return loss_array


# In[8]:


def validate_epoch(model, test_loader, loss_function):
    model.train(False)
    running_loss = 0.

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch.squeeze())
            running_loss += loss.item()

    avg_loss = running_loss / len(test_loader)
    
    log_file = open(sys.argv[3], "a")
    log_file.write("Val loss: {0:.5f}\n".format(avg_loss))
    log_file.close()
    
    log_file = open(sys.argv[3], "a")
    log_file.write("_______________________________\n")
    log_file.close()
    
    log_file = open(sys.argv[3], "a")
    log_file.write("\n")
    log_file.close()


    return avg_loss


log_file = open(sys.argv[3], "a")
log_file.write("Definitions complete\n")
log_file.close()

log_file = open(sys.argv[3], "a")
log_file.write("\n")
log_file.close()

# In[9]:


train_sample = np.random.choice(list(xr_train.keys()), size=70, replace=False)
test_sample = np.random.choice(list(xr_test.keys()), size=30, replace=False)

xr_train_sample = xr_train[train_sample].copy()
xr_test_sample = xr_test[test_sample].copy()

xr_train_sample = xr_train_sample.assign_coords({"date_sin": xr_train.date_sin.data,
                    "date_cos": xr_train.date_cos.data,
                    "time_sin": xr_train.time_sin.data,
                    "time_cos": xr_train.time_cos.data}).copy()

xr_test_sample = xr_test_sample.assign_coords({"date_sin": xr_test.date_sin.data,
                    "date_cos": xr_test.date_cos.data,
                    "time_sin": xr_test.time_sin.data,
                    "time_cos": xr_test.time_cos.data}).copy()

df_distances_train_sample = df_distances_train[train_sample].loc[train_sample].copy()
df_distances_test_sample = df_distances_test[test_sample].loc[test_sample].copy()

log_file = open(sys.argv[3], "a")
log_file.write("Sample ready\n")
log_file.close()

log_file = open(sys.argv[3], "a")
log_file.write("\n")
log_file.close()


def objective(trial):
    n_neighbours = 0
    cut_off_km = 0
    with_neighbours = False
    
    num_layers = trial.suggest_int('num_layers', 2, 5)
    hidden_size = 2**trial.suggest_int('hidden_size', 6, 8)
    look_up = 6*trial.suggest_int('look_up', 1, 4)
    dropout = trial.suggest_float('dropout', 0, 1)

    data_params = {
    'xr_test': xr_test_sample,
    'xr_train': xr_train_sample,
    'batch_size': 128,
    'dict_look_up': {'year': 0, 'month': 0, 'week': 0, 'day': 0, 'hour': look_up},
    'dict_look_ahead': {'year': 0, 'month': 0, 'week': 0, 'day': 0, 'hour': 6},
    'cut_off_km': cut_off_km,
    'n_neighbours': n_neighbours,
    'df_metadata_train': df_metadata_train,
    'df_metadata_test': df_metadata_test,
    'with_neighbours': with_neighbours,
    'era': era
    }

    train_loader, test_loader, look_ahead = loaders(**data_params)

    model_params = {
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'out_features': look_ahead,
        'input_size': n_neighbours+1+4+3,
        'dropout': dropout
    }
    
    log_file = open(sys.argv[3], "a")
    log_file.write("model_params: " + str(model_params)+"\n")
    log_file.close()


    model = LSTM_model(**model_params)

    model.to(device)

    learning_rate = 0.001
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 6

    train_loss = []
    test_loss = []

    best_loss = 1
    best_epoch = 0
    
    model_time = time.time()

    for epoch in range(num_epochs):
        # hard stop if training time over 1.5 hours
        if (time.time()-model_time) > 5400:
            break        
        log_file = open(sys.argv[3], "a")
        log_file.write("Epoch: " + str(epoch+1)+"\n")
        log_file.close()

        train_loss.append(train_epoch(model=model, train_loader=train_loader,
                                    optimizer=optimizer,
                                    loss_function=loss_function,
                                    print_every=300))
        test_loss.append(validate_epoch(model=model, test_loader=test_loader, loss_function=loss_function))
        if test_loss[-1] < best_loss:
            best_loss = test_loss[-1]
            best_epoch = epoch
        if epoch - best_epoch >= 5:
            break

    return best_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=n_trials)

trial = study.best_trial

log_file = open(sys.argv[3], "a")
log_file.write("Accuracy: {}\n".format(trial.value))
log_file.close()

log_file = open(sys.argv[3], "a")
log_file.write("Best hyperparameters: {}\n".format(trial.params))
log_file.close()



# In[ ]:


best_params_df = pd.DataFrame(data=trial.params, index=[0])

best_params_df.at[0,'hidden_size'] = 2**best_params_df.loc[0]['hidden_size'] 
best_params_df.at[0,'look_up'] = 6*best_params_df.loc[0]['look_up']  

best_params_df.to_csv('era_best_params.csv', index=False)
log_file = open(sys.argv[3], "a")
log_file.write("Best params saved\n")
log_file.close()

log_file = open(sys.argv[3], "a")
log_file.write("END LOG\n")
log_file.close()

