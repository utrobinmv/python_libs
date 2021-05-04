import numpy as np

from catboost import CatBoostClassifier, Pool
from catboost import CatBoostRegressor
from catboost import CatBoost


# initialize data
train_data = np.random.randint(0,
                               100, 
                               size=(100, 10))

train_labels = np.random.randint(0,
                                 2,
                                 size=(100))

test_data = catboost_pool = Pool(train_data, 
                                 train_labels)

model = CatBoostClassifier(iterations=2,
                           depth=2,
                           learning_rate=1,
                           loss_function='Logloss',
                           verbose=True)
# train the model
model.fit(train_data, train_labels)
# make the prediction using the resulting model
preds_class = model.predict(test_data)
preds_proba = model.predict_proba(test_data)
print("class = ", preds_class)
print("proba = ", preds_proba)



train_data = np.random.randint(0, 
                               100, 
                               size=(100, 10))
train_label = np.random.randint(0, 
                                1000, 
                                size=(100))
test_data = np.random.randint(0, 
                              100, 
                              size=(50, 10))
# initialize Pool
train_pool = Pool(train_data, 
                  train_label, 
                  cat_features=[0,2,5])
test_pool = Pool(test_data, 
                 cat_features=[0,2,5]) 

# specify the training parameters 
model = CatBoostRegressor(iterations=2, 
                          depth=2, 
                          learning_rate=1, 
                          loss_function='RMSE')
#train the model
model.fit(train_pool)
# make the prediction using the resulting model
preds = model.predict(test_pool)
print(preds)


# read the dataset

train_data = np.random.randint(0, 
                               100, 
                               size=(100, 10))
train_labels = np.random.randint(0, 
                                2, 
                                size=(100))
test_data = np.random.randint(0, 
                                100, 
                                size=(50, 10))
                                
train_pool = Pool(train_data, 
                  train_labels)

test_pool = Pool(test_data) 
# specify training parameters via map

param = {'iterations':5}
model = CatBoost(param)
#train the model
model.fit(train_pool) 
# make the prediction using the resulting model
preds_class = model.predict(test_pool, prediction_type='Class')
preds_proba = model.predict(test_pool, prediction_type='Probability')
preds_raw_vals = model.predict(test_pool, prediction_type='RawFormulaVal')
print("Class", preds_class)
print("Proba", preds_proba)
print("Raw", preds_raw_vals)


#CatBoost grid_search

param_grid={'l2_leaf_reg': np.linspace(0, 1, 20)}
my_model = catboost.CatBoostClassifier(n_estimators=200, silent=True,eval_metric='AUC')
grid_search_result = my_model.grid_search(param_grid, X, y, plot=True, refit=True)
grid_search_result['params']



#Handmade grid_search CatBoost
from catboost import CatBoostClassifier, Pool

X = data[feature_cols]
y = data[target_col]

X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)
eval_dataset = Pool(X_eval, y_eval, cat_features=cat_cols)
best_parametrs = {'best_lr': 0, 'best_depth': 0}
best_auc_score = 0

# Brute Force parametrs
for curr_depth in np.arange(3, 9, 1):
  for curr_lr in np.arange(0.01, 0.3, 0.02):
    new_clf = CatBoostClassifier(learning_rate = curr_lr, custom_metric=['AUC'],
                                 depth = curr_depth, cat_features=cat_cols)
    new_clf.fit(X_train, y_train, eval_set=eval_dataset,
          verbose=False)
    if new_clf.get_best_score()['validation']['AUC'] > best_auc_score:
      print(new_clf.get_best_score()['validation']['AUC'])
      print("Updated best score")
      best_auc_score = new_clf.get_best_score()['validation']['AUC']
      best_parametrs = {
          'best_lr': curr_lr,
          'best_depth': curr_depth
      }
      print(best_parametrs)
      
print(best_parametrs)
print(best_auc_score)
