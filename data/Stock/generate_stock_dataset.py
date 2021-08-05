import numpy as np

np.random.seed(1000)

from scipy.io import loadmat

stock = r""
year = "2015"
MAX_TIMESTEPS_LIST = []

# for j in range(20):
''' Load train set '''
train_data = loadmat(stock + "training_set_stock_" + str(year) + ".mat")
test_data = loadmat(stock + "test_set_stock_" + str(year) + ".mat")
X_train_mat = train_data['training_set'][0]
y_train_mat = train_data['train_labels'][0]
X_test_mat = test_data['test_set'][0]
y_test_mat = test_data['test_labels'][0]

# add all data
# train_data = loadmat(stock + "original_domain_training_set.mat")
# test_data = loadmat(stock + "original_domain_test_set.mat")
# X_train_mat = train_data['original_training_data_set'][0]
# y_train_mat = train_data['original_training_data_label'][0]
# X_test_mat = test_data['original_test_data_set'][0]
# y_test_mat = test_data['original_test_data_label'][0]

y_train = y_train_mat.reshape(-1, 1)
y_test = y_test_mat.reshape(-1, 1)
# 求每个序列的长度
var_list = []
for i in range(X_train_mat.shape[0]):
    var_count = X_train_mat[i].shape[-1]
    var_list.append(var_count)

var_list = np.array(var_list)
# max_nb_timesteps = var_list.max()
# min_nb_timesteps = var_list.min()
max_nb_timesteps = 720
min_nb_timesteps = 10
median_nb_timesteps = np.median(var_list)

print('max nb timesteps train : ', max_nb_timesteps)
MAX_TIMESTEPS_LIST.append(max_nb_timesteps)
print('min nb timesteps train : ', min_nb_timesteps)
print('median_nb_timesteps nb timesteps train : ', median_nb_timesteps)

# X_train = np.zeros((X_train_mat.shape[0], X_train_mat[0].shape[0], 1203))
X_train = np.zeros((X_train_mat.shape[0], X_train_mat[0].shape[0], max_nb_timesteps))

# pad ending with zeros to get numpy arrays
for i in range(X_train_mat.shape[0]):
    var_count = X_train_mat[i].shape[-1]
    var_count = min(720, var_count)
    # print(i, X_train_mat[i])
    X_train[i, :, :var_count] = X_train_mat[i][:, :var_count]

# ''' Load test set '''

# X_test = np.zeros((X_test_mat.shape[0], X_test_mat[0].shape[0], 1203))
X_test = np.zeros((X_test_mat.shape[0], X_test_mat[0].shape[0], max_nb_timesteps))

# pad ending with zeros to get numpy arrays
for i in range(X_test_mat.shape[0]):
    var_count = X_test_mat[i].shape[-1]
    X_test[i, :, :var_count] = X_test_mat[i][:, :max_nb_timesteps]

# ''' Save the datasets '''
print("Train dataset : ", X_train.shape, y_train.shape)
print("Test dataset : ", X_test.shape, y_test.shape)
print("Train dataset metrics : ", X_train.mean(), X_train.std())
print("Test dataset : ", X_test.mean(), X_test.std())
print("Nb classes : ", len(np.unique(y_train)))

np.save(stock + 'X_train.npy', X_train)
np.save(stock + 'y_train.npy', y_train)
np.save(stock + 'X_test.npy', X_test)
np.save(stock + 'y_test.npy', y_test)

