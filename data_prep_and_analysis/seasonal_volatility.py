xr_train = xr.open_dataset(path+'xr_train_timestamped.netcdf')
xr_test = xr.open_dataset(path+'xr_test_timestamped.netcdf')

df_metadata_train = pd.read_csv(path + 'df_metadata_train.csv', index_col=0)
df_metadata_train.index = df_metadata_train.index.astype(str)

df_metadata_test = pd.read_csv(path + 'df_metadata_test.csv', index_col=0)
df_metadata_test.index = df_metadata_test.index.astype(str)

season_idxs=xr_train.groupby('datetime.season').groups

def mean_abs_step(arr):
  steps = abs(arr[1:] - arr[:-1])
  return np.mean(steps)

def MAS_column(xr, idx):
  col = np.zeros(len(idx))
  for i, ss_id in enumerate(idx):
    col[i] = mean_abs_step(xr[ss_id].data)
  return col

def std_column(xr, idx):
  col = np.zeros(len(idx))
  for i, ss_id in enumerate(idx):
    col[i] = np.std(xr[ss_id].data)
  return col

df_metadata_train['full_std'] = std_column(xr_train, df_metadata_train.index.values)
df_metadata_test['full_std'] = std_column(xr_test, df_metadata_test.index.values)

for key in list(season_idxs.keys()):
  df_metadata_train['std_'+key] = std_column(xr_train.isel(datetime=season_idxs[key]), df_metadata_train.index.values)
  df_metadata_test['std_'+key] = std_column(xr_test.isel(datetime=season_idxs[key]), df_metadata_test.index.values)

df_metadata_test['full_MAS'] = MAS_column(xr_test, df_metadata_test.index.values)

for key in list(season_idxs.keys()):
  df_metadata_train['full_'+key] = MAS_column(xr_train.isel(datetime=season_idxs[key]), df_metadata_train.index.values)
  df_metadata_test['full_'+key] = MAS_column(xr_test.isel(datetime=season_idxs[key]), df_metadata_test.index.values)
