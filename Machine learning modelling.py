#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

print('Numpy version:', np.__version__)
print('Pandas version:', pd.__version__)


# In[2]:


import sklearn as sk
sk.__version__


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


# import dataset

# In[62]:


data_train = pd.read_excel(r'E:/R code-LMX/R code for SEER for publishing paper/Data/start0.8-number.xlsx')
data_test = pd.read_excel(r'E:/R code-LMX/R code for SEER for publishing paper/Data/starv0.2-number.xlsx')


# In[63]:


pd.set_option('display.max_columns', data_train.shape[1])
pd.set_option('max_colwidth', 1000)


# In[64]:


data_train.head()


# In[65]:


data_train.info()


# Data preprocessing piprlines
# Prepare the data to a format that can be fit into scikit learn algorithms

# Categorical variable encoder

# In[66]:


categorical_vars = ['primarysite', 'Histology', 'race', 'Sex', 'tstage', 'nstage', 'brainm', 'liverm', 'surgery', 'Radiation', 'Chemotherapy']


# In[67]:


data_train[categorical_vars].head()


# In[68]:


# to make a custom transformer to fit into a pipeline
class Vars_selector(BaseEstimator, TransformerMixin):
    '''Returns a subset of variables in a dataframe'''
    def __init__(self, var_names):
        '''var_names is a list of categorical variables names'''
        self.var_names = var_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        '''returns a dataframe with selected variables'''
        return X[self.var_names]


# In[69]:


class Cat_vars_encoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        '''X is a dataframe'''

        return X.values


# Transform data in a pipeline

# In[70]:


# categorical variables preprocessing
cat_vars_pipeline = Pipeline([
    ('selector', Vars_selector(categorical_vars)),
    ('encoder', Cat_vars_encoder())
])


# For many machine learning algorithms, gradient descent is the preferred or even the only optimization method to learn the model parameters. Gradient descent is highly sensitive to feature scaling.

# ** Continuous vars **

# In[71]:


continuous_vars = ['age']


# In[72]:


data_train[continuous_vars].describe()


# The scales among the continuous variables vary a lot, we need to standardize them prior to modelling.

# In[73]:


# continuous variables preprocessing
cont_vars_pipeline = Pipeline([
    ('selector', Vars_selector(continuous_vars)),
    ('standardizer', StandardScaler())
])


# To transform the two types of variables in one step

# In[74]:


preproc_pipeline = FeatureUnion(transformer_list=[
    ('cat_pipeline', cat_vars_pipeline),
    ('cont_pipeline', cont_vars_pipeline)
])


# In[75]:


data_train_X = pd.DataFrame(preproc_pipeline.fit_transform(data_train), 
                            columns=categorical_vars + continuous_vars)


# In[76]:


data_train_X.head()


# Fitting classifiers

# In[77]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import learning_curve, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score


# In[78]:


data_train['anxietyanddepression'].value_counts()


# This is a fairly balanced dataset(i.e., the number of positive and negative cases are roughly the same), and we'll use AUC as our metric to optimise the model performance.

# Assessing learning curve using the model default settings
# Tuning the model hyper-parameters are always difficult, so a good starting point is to see how the Scikit-learn default settings for the model performs, i.e., to see if it overfits or underfits, or is just right. This will give a good indication as to the direction of tuning.

# In[79]:


def plot_learning_curves(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(model, X, y, cv=5, scoring='roc_auc',
                                                           random_state=42, n_jobs=-1)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), "o-", label="training scores")
    plt.plot(train_sizes, np.mean(val_scores, axis=1), "o-", label="x-val scores")
    plt.legend(fontsize=14).get_frame().set_facecolor('white')
    plt.xlabel("Training set size")
    plt.ylabel("Area under Curve")
    plt.title('{} learning curve'.format(model.__class__.__name__))


# In[ ]:





# # Logistic Regression---网格搜索模型调参GridSearchCV

# In[22]:


lr_clf = LogisticRegression(n_jobs = -1)
plot_learning_curves(lr_clf, data_train_X, data_train['anxietyanddepression'])


# In[24]:


ra_plot = lr_clf.shap_summary_plot()


# In[ ]:





# Let's see if we can squeeze some more performance out by optimising C

# In[183]:


param_grid = {
        'C': [0.1, 1, 10],
    }
lr_clf = LogisticRegression(random_state=42)
grid_search = GridSearchCV(lr_clf, param_grid=param_grid, return_train_score=True,
                                cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(data_train_X, data_train['anxietyanddepression'])


# In[184]:


cv_rlt = grid_search.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# # Looks like C=100? is our best value.

# 下方with open,"wb",确实是保存了新的pkl模型文件，下一次使用该文件只需要直接调用即可

# In[185]:


lr_clf = grid_search.best_estimator_
with open('E:/R code-LMX/R code for SEER for publishing paper/Data/lr_clf_final_round.pkl', 'wb') as f:
    pickle.dump(lr_clf, f)


# In[186]:


plot_learning_curves(lr_clf, data_train_X, data_train['anxietyanddepression'])


# Looks like the logistic regression model would benefit from additional data.

# # XGboot

# In[22]:


from xgboost.sklearn import XGBClassifier


# In[23]:


Xgbc_clf=XGBClassifier(random_state=42)  #Xgbc
plot_learning_curves(Xgbc_clf, data_train_X, data_train['anxietyanddepression'])


# max_depth = 5 ：这应该在3-10之间。我从5开始，但你也可以选择不同的数字。4-6可以是很好的起点。
# min_child_weight = 1 ：选择较小的值是因为它是高度不平衡的类问题，并且叶节点可以具有较小的大小组。
# gamma = 0.1 ：也可以选择较小的值，如0.1-0.2来启动。无论如何，这将在以后进行调整。
# subsample，colsample_bytree = 0.8：这是一个常用的使用起始值。典型值介于0.5-0.9之间。
# scale_pos_weight = 1：由于高级别的不平衡。
# colsample_bytree = 0.5,gamma=0.2

# In[290]:


param_distribs = {
     'n_estimators': stats.randint(low=60, high=120),      
    'max_depth': stats.randint(low=1, high=10),
    'min_child_weight': stats.randint(low=1, high=10)
    }
Xgbc_clf=XGBClassifier(random_state=42,learning_rate=0.125,use_label_encoder=False)
Xgbc_search = RandomizedSearchCV(Xgbc_clf, param_distributions=param_distribs, return_train_score=True,
                                n_iter=100, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
Xgbc_gs=Xgbc_search.fit(data_train_X, data_train['anxietyanddepression'])


# In[291]:


print(Xgbc_gs.best_score_)


# In[292]:


print(Xgbc_gs.best_params_)


# In[293]:


cv_rlt = Xgbc_search.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# In[294]:


rf_clf = Xgbc_search.best_estimator_
rf_clf.fit(data_train_X, data_train['anxietyanddepression'])
with open('E:/R code-LMX/R code for SEER for publishing paper/Data/Xgbc_clf_final_round.pkl', 'wb') as f:
    pickle.dump(rf_clf, f)


# In[295]:


rf_clf


# In[296]:


plot_learning_curves(rf_clf, data_train_X, data_train['anxietyanddepression'])


# In[297]:


# Import model and retrain
with open('E:/R code-LMX/R code for SEER for publishing paper/Data/Xgbc_clf_final_round.pkl', 'rb') as f:
    Xgbc_clf = pickle.load(f)
Xgbc_clf.fit(data_train_X, data_train['anxietyanddepression'])


# In[298]:


accu_Xgbc = accuracy_score(data_test['anxietyanddepression'], Xgbc_clf.predict(data_test_X))
round(accu_Xgbc,3)


# In[299]:


pd.crosstab(data_test['anxietyanddepression'], Xgbc_clf.predict(data_test_X))


# In[300]:


pred_proba_Xgbc = Xgbc_clf.predict_proba(data_test_X)


# In[301]:


fpr, tpr, _ = roc_curve(data_test['anxietyanddepression'], pred_proba_Xgbc[:, 1])
auc_Xgbc = roc_auc_score(data_test['anxietyanddepression'], pred_proba_Xgbc[:, 1])
round(auc_Xgbc,3)


# In[302]:


plot_roc_curve(fpr, tpr, round(auc_Xgbc,3), Xgbc_clf)


# In[303]:


data_test['lr_pred_proba'] = pred_proba_Xgbc[:, 1]
data_test.to_csv('E:/R code-LMX/R code for SEER for publishing paper/Data/test_set_with_predictions-Xgbc.csv'.format(len(data_train)), index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# # DecisionTreeClassifier

# In[384]:


from sklearn.tree import DecisionTreeClassifier


# In[385]:


tr_clf=DecisionTreeClassifier(random_state=42)  # 决策树模型
plot_learning_curves(tr_clf, data_train_X, data_train['anxietyanddepression'])


# In[516]:


param_distribs = {
         'max_features': ['auto', 'log2'],
        'max_depth': stats.randint(low=1, high=50),
        'min_samples_split': stats.randint(low=2, high=1000), 
        'min_samples_leaf': stats.randint(low=2, high=1000)
    }
dt_clf = DecisionTreeClassifier(random_state=42,criterion='gini', splitter='best')
rnd_search = RandomizedSearchCV(dt_clf, param_distributions=param_distribs, return_train_score=True,
                                n_iter=100, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
gsdt=rnd_search.fit(data_train_X, data_train['anxietyanddepression'])


# In[517]:


print(gsdt.best_score_)


# In[518]:


print(gsdt.best_params_)


# In[519]:


cv_rlt = rnd_search.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# In[520]:


dt_clf = rnd_search.best_estimator_
dt_clf.fit(data_train_X, data_train['anxietyanddepression'])
with open('E:/R code-LMX/R code for SEER for publishing paper/Data/dt_clf_final_round.pkl', 'wb') as f:
    pickle.dump(dt_clf, f)


# In[521]:


plot_learning_curves(dt_clf, data_train_X, data_train['anxietyanddepression'])


# In[ ]:





# In[522]:


# Import model and retrain
with open('E:/R code-LMX/R code for SEER for publishing paper/Data/dt_clf_final_round.pkl', 'rb') as f:
    dt_clf = pickle.load(f)
dt_clf.fit(data_train_X, data_train['anxietyanddepression'])


# In[523]:


accu_dt = accuracy_score(data_test['anxietyanddepression'], dt_clf.predict(data_test_X))
round(accu_dt,3)


# In[524]:


pd.crosstab(data_test['anxietyanddepression'], dt_clf.predict(data_test_X))


# In[525]:


pred_proba_dt = dt_clf.predict_proba(data_test_X)
fpr, tpr, _ = roc_curve(data_test['anxietyanddepression'], pred_proba_dt[:, 1])
auc_dt = roc_auc_score(data_test['anxietyanddepression'], pred_proba_dt[:, 1])
round(auc_dt,3)


# In[526]:


plot_roc_curve(fpr, tpr, round(auc_dt,3), dt_clf)


# In[527]:


data_test['lr_pred_proba'] = pred_proba_dt[:, 1]


# In[528]:


data_test.to_csv('E:/R code-LMX/R code for SEER for publishing paper/Data/test_set_with_predictions-decision tree.csv'.format(len(data_train)), index=False)


# In[ ]:





# # Random Forests classifier---随机搜索模型调参RandomizedSearchCV
# Random forests classifier is an ensemble tree-based model that reduces the variance of the predictors.
# 
# plot the learning curve to find out where the default model is at

# In[ ]:





# In[304]:


rf_clf = RandomForestClassifier(random_state=42)
plot_learning_curves(rf_clf, data_train_X, data_train['anxietyanddepression'])


# In[564]:


param_distribs = {
        'n_estimators': stats.randint(low=1, high=50),
         'max_features': ['auto', 'log2'],
        'max_depth': stats.randint(low=1, high=5),
        'min_samples_split': stats.randint(low=2, high=100), 
        'min_samples_leaf': stats.randint(low=2, high=100)
    }
rf_clf = RandomForestClassifier(random_state=42)
rnd_search = RandomizedSearchCV(rf_clf, param_distributions=param_distribs, return_train_score=True,
                                n_iter=100, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
gs=rnd_search.fit(data_train_X, data_train['anxietyanddepression'])


# In[565]:


print(gs.best_score_)


# In[566]:


print(gs.best_params_)


# In[567]:


cv_rlt = rnd_search.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# In[568]:


rf_clf = rnd_search.best_estimator_
rf_clf.fit(data_train_X, data_train['anxietyanddepression'])
with open('E:/R code-LMX/R code for SEER for publishing paper/Data/rf_clf_final_round.pkl', 'wb') as f:
    pickle.dump(rf_clf, f)


# In[569]:


plot_learning_curves(rf_clf, data_train_X, data_train['anxietyanddepression'])


# In[34]:





# 神经网络

# In[35]:


from sklearn.neural_network import MLPClassifier


# In[36]:


nn_clf = MLPClassifier(random_state=42)
plot_learning_curves(nn_clf, data_train_X, data_train['anxietyanddepression'])


# In[73]:


nn_clf = MLPClassifier(random_state=42,activation='relu',alpha=0.0001,batch_size='auto',beta_1=0.9, beta_2=0.999, 
                       early_stopping=False,epsilon=1e-08,hidden_layer_sizes=(100),
                       learning_rate='constant', learning_rate_init=0.001,max_iter=200, momentum=0.9, 
                       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,shuffle=True,  
                       tol=0.0001, validation_fraction=0.1,verbose=False, warm_start=False)


# In[78]:


nn_clf.fit(data_train_X, data_train['anxietyanddepression'])
nn_clf_y_pre=nn_clf.predict(data_test_X)
nn_clf_y_proba=nn_clf.predict_proba(data_test_X)


# In[79]:


from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score,roc_curve


# In[80]:


nn_clf_accuracy_score=accuracy_score(data_test['anxietyanddepression'],svm_y_pre)
nn_clf_preci_score=precision_score(data_test['anxietyanddepression'],svm_y_pre)
nn_clf_recall_score=recall_score(data_test['anxietyanddepression'],svm_y_pre)
nn_clf_f1_score=f1_score(data_test['anxietyanddepression'],svm_y_pre)
nn_clf_auc=roc_auc_score(data_test['anxietyanddepression'],svm_y_proba[:,1])
print('nn_clf_accuracy_score: %f,nn_clf_preci_score: %f,nn_clf_recall_score: %f,nn_clf_f1_score: %f,nn_clf_auc: %f'
      %(nn_clf_accuracy_score,nn_clf_preci_score,nn_clf_recall_score,nn_clf_f1_score,nn_clf_auc))


# In[81]:


nn_clf_fpr,nn_clf_tpr,nn_clf_threasholds=roc_curve(data_test['anxietyanddepression'],nn_clf_y_proba[:,1]) # 计算ROC的值,svm_threasholds为阈值
plt.title("roc_curve of %s(AUC=%.4f)" %('nn_clf',nn_clf_auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(nn_clf_fpr,nn_clf_tpr)
plt.show()


# In[82]:


data_test['lr_pred_proba'] = svm_y_proba[:,1]


# In[83]:


data_test.to_csv('E:/R code-LMX/R code for SEER for publishing paper/Data/test_set_with_predictions-nn.csv'.format(len(data_train)), index=False)


# In[ ]:





# # Gradient boosting classifier---随机搜索模型调参RandomizedSearchCV
# Gradient boosting classifier is an ensemble tree-based model that reduces the bias of the predictors.

# In[369]:


plot_learning_curves(GradientBoostingClassifier(random_state=42), data_train_X, data_train['anxietyanddepression'])


# In[370]:



param_distribs = {
        'n_estimators': stats.randint(low=80, high=200),
         'max_features': ['auto', 'log2'],
        'max_depth': stats.randint(low=1, high=50),
        'min_samples_split': stats.randint(low=2, high=10), 
        'min_samples_leaf': stats.randint(low=2, high=10),
    }

rnd_search = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), 
                                param_distributions=param_distribs, return_train_score=True,
                                n_iter=100, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
# this will take a long time
gsgbm = rnd_search.fit(data_train_X, data_train['anxietyanddepression'])


# In[149]:


print(gsgbm.best_score_)


# In[150]:


print(gsgbm.best_params_)


# In[151]:


cv_rlt = rnd_search.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# In[153]:


gbm_clf = rnd_search.best_estimator_
gbm_clf.fit(data_train_X, data_train['anxietyanddepression'])
with open('E:/R code-LMX/R code for SEER for publishing paper/Data/gbm_clf_final_round.pkl', 'wb') as f:
    pickle.dump(gbm_clf, f)


# In[154]:


plot_learning_curves(gbm_clf, data_train_X, data_train['anxietyanddepression'])


# In[ ]:





# # Support vector machine classifier---网格搜索模型调参GridSearchCV
# Support vector machine classifier is a powerful classifier that works best on small to medium size complex data set. Our training set is medium size to SVMs.
# 
# plot the learning curve to find out where the default model is at

# In[ ]:





# Try Linear SVC fist

# In[36]:


plot_learning_curves(LinearSVC(loss='hinge', random_state=42), data_train_X, data_train['anxietyanddepression'])


# Try Polynomial kernel

# In[38]:


plot_learning_curves(SVC(kernel='poly', random_state=42), data_train_X, data_train['anxietyanddepression'])


# Try Gaussian RBF kernel

# In[22]:


plot_learning_curves(SVC(random_state=42), data_train_X, data_train['anxietyanddepression'])


# In[1]:


rbf_gamma = [1/len(data_train_X.columns) * x for x in range(1, 15, 5)]
param_grid = [
    # first try Poly kernel
    ## coef0 hyper-parameter was also tested originally, however, it is taking too long
    {'kernel':['poly'], 'degree': [3, 9, 15], 'C': [1, 3, 9]},
    # then try RBF kernel
    {'gamma': rbf_gamma, 'C': [1, 3, 9]},
  ]

grid_search = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=5,
                           scoring='roc_auc', n_jobs=-1)

# this will take a long time
grid_search.fit(data_train_X, data_train['anxietyanddepression'])


# In[ ]:


cv_rlt = grid_search.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# In[ ]:


svc_clf = grid_search.best_estimator_
svc_clf.fit(data_train_X, data_train['anxietyanddepression'])
with open('E:/R code-LMX/R code for SEER for publishing paper/Data/svc_clf_final_round.pkl', 'wb') as f:
    pickle.dump(svc_clf, f)


# In[ ]:


# best model is the default RBF kernal SVM
plot_learning_curves(svc_clf, data_train_X, data_train['anxietyanddepression']) 


# In[ ]:





# In[ ]:





# 0. 不调参数(seer-lung cancer使用)

# In[ ]:


svm = SVC(kernel='poly',probability=True,random_state=2018,tol=1e-6)  # SVM模型
svm.fit(data_train_X, data_train['anxietyanddepression'])
svm_y_pre=svm.predict(data_test_X)
svm_y_proba=svm.predict_proba(data_test_X)


# In[27]:


from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score,roc_curve


# In[28]:


svm_accuracy_score=accuracy_score(data_test['anxietyanddepression'],svm_y_pre)
svm_preci_score=precision_score(data_test['anxietyanddepression'],svm_y_pre)
svm_recall_score=recall_score(data_test['anxietyanddepression'],svm_y_pre)
svm_f1_score=f1_score(data_test['anxietyanddepression'],svm_y_pre)
svm_auc=roc_auc_score(data_test['anxietyanddepression'],svm_y_proba[:,1])
print('svm_accuracy_score: %f,svm_preci_score: %f,svm_recall_score: %f,svm_f1_score: %f,svm_auc: %f'
      %(svm_accuracy_score,svm_preci_score,svm_recall_score,svm_f1_score,svm_auc))


# In[30]:


svm_fpr,svm_tpr,svm_threasholds=roc_curve(data_test['anxietyanddepression'],svm_y_proba[:,1]) # 计算ROC的值,svm_threasholds为阈值
plt.title("roc_curve of %s(AUC=%.4f)" %('svm',svm_auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(svm_fpr,svm_tpr)
plt.show()


# In[31]:


data_test['lr_pred_proba'] = svm_y_proba[:,1]


# In[32]:


data_test.to_csv('E:/R code-LMX/R code for SEER for publishing paper/Data/test_set_with_predictions-Support vector machine model2.csv'.format(len(data_train)), index=False)


# In[ ]:





# In[ ]:


svc_clf = grid.best_estimator_
#svc_clf.fit(data_train_X, data_train['anxietyanddepression'])
with open('E:/R code-LMX/R code for SEER for publishing paper/Data/svc_clf_final_round.pkl', 'wb') as f:
    pickle.dump(svc_clf, f)


# In[ ]:





# In[ ]:





# 第一调参

# In[ ]:


hyperparameters = {
 'C': [0.1, 1, 100, 1000],
 'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5],
 'kernel': ('linear', 'rbf')
}
grid = GridSearchCV(
 estimator=SVC(probability=True),
 param_grid=hyperparameters,
 cv=5, return_train_score=True,
scoring='f1_micro', 
n_jobs=-1)
gssvm = grid.fit(data_train_X, data_train['anxietyanddepression'])


# In[202]:


print(gssvm.best_score_)


# In[128]:


print(gssvm.best_params_)


# In[129]:


cv_rlt = grid.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# In[130]:


svc_clf = grid.best_estimator_
#svc_clf.fit(data_train_X, data_train['anxietyanddepression'])
with open('E:/R code-LMX/R code for SEER for publishing paper/Data/svc_clf_final_round.pkl', 'wb') as f:
    pickle.dump(svc_clf, f)


# In[131]:


svc_clf


# In[132]:


# best model is the default RBF kernal SVM
plot_learning_curves(svc_clf, data_train_X, data_train['anxietyanddepression']) 


# In[ ]:





# 第二调参

# In[530]:


hyperparameters = {
 "C": stats.uniform(0.001, 0.1),
 "gamma": stats.uniform(0, 0.5),
 'kernel': ('linear', 'rbf')
}
random = RandomizedSearchCV(estimator = SVC(probability=True), param_distributions = hyperparameters, n_iter = 100, 
                            cv = 5, return_train_score=True, random_state=42, n_jobs = -1)
gssvm = random.fit(data_train_X, data_train['anxietyanddepression'])


# In[531]:


print(gssvm.best_score_)


# In[532]:


print(gssvm.best_params_)


# In[533]:


cv_rlt = random.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# In[536]:


svc_clf = random.best_estimator_
#svc_clf.fit(data_train_X, data_train['anxietyanddepression'])
with open('E:/R code-LMX/R code for SEER for publishing paper/Data/svc_clf_final_round.pkl', 'wb') as f:
    pickle.dump(svc_clf, f)


# In[535]:


# best model is the default RBF kernal SVM
plot_learning_curves(svc_clf, data_train_X, data_train['anxietyanddepression']) 


# In[ ]:





# 第三调参--不用

# In[94]:


rbf_gamma = [1/len(data_train_X.columns) * x for x in range(1, 15, 5)]
param_grid = [
    # first try Poly kernel
    ## coef0 hyper-parameter was also tested originally, however, it is taking too long
    {'kernel':['poly'], 'degree': [3, 9, 15], 'C': [1, 3, 9]},
    # then try RBF kernel
    {'gamma': rbf_gamma, 'C': [1, 3, 9]},
  ]

grid_search = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=5,
                           scoring='roc_auc', n_jobs=-1)

# this will take a long time
grid_search.fit(data_train_X, data_train['anxietyanddepression'])


# In[95]:


cv_rlt = grid_search.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# In[96]:


cv_rlt = grid_search.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"],  cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores',  'Params'])


# In[97]:


svc_clf = grid_search.best_estimator_
svc_clf.fit(data_train_X, data_train['anxietyanddepression'])
with open('E:/svc_clf_final_round.pkl', 'wb') as f:
    pickle.dump(svc_clf, f)


# In[98]:


# best model is the default RBF kernal SVM
plot_learning_curves(svc_clf, data_train_X, data_train['anxietyanddepression']) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Ensemble classifier
# Scikit-learn offers a voting classifier which aggregates the prediction of multiple predictors and is a flexible ensemble technique that allows an ensemble of different models.
# For the final classifier, simply aggregate the predictions of the three best models, i.e., random forests, gradien boosting machine and the support vector machine.

# In[548]:


ensemble_clf = VotingClassifier(estimators=[('rf', rf_clf), ('gbm', gbm_clf), ('svc', svc_clf),('dt',dt_clf)],
                             voting='soft')
ensemble_clf.fit(data_train_X, data_train['anxietyanddepression'])


# Check out its learning curve.

# In[549]:


plot_learning_curves(ensemble_clf, data_train_X, data_train['anxietyanddepression'])


# In[550]:


with open('E:/R code-LMX/R code for SEER for publishing paper/Data/ensemble_clf_final_round.pkl', 'wb') as f:
    pickle.dump(ensemble_clf, f)


# In[ ]:





# In[ ]:





# # h2o AutoML

# In[28]:


import h2o
from h2o.automl import H2OAutoML


# In[ ]:





# In[91]:


import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator


# In[92]:


h2o.init()


# In[99]:


data_train = h2o.import_file(r'E:/R code-LMX/R code for SEER for publishing paper/Data/start0.8.csv')##只能是csv文件，excle文件还不行哦！！！
data_test = h2o.import_file(r'E:/R code-LMX/R code for SEER for publishing paper/Data/starv0.2.csv')


# In[97]:


airlines = h2o.import_file('https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip')


# In[100]:


data_train.head()


# In[58]:





# In[111]:


data_train['primarysite'] = data_train['primarysite'].asfactor()
data_train['Histology'] = data_train['Histology'].asfactor()
data_train['race'] = data_train['race'].asfactor()
data_train['Sex'] = data_train['Sex'].asfactor()
data_train['tstage'] = data_train['tstage'].asfactor()
data_train['nstage'] = data_train['nstage'].asfactor()
data_train['brainm'] = data_train['brainm'].asfactor()
data_train['liverm'] = data_train['liverm'].asfactor()
data_train['surgery'] = data_train['surgery'].asfactor()
data_train['Radiation'] = data_train['Radiation'].asfactor()
data_train['Chemotherapy'] = data_train['Chemotherapy'].asfactor()


# In[112]:


predictors = ['age','primarysite', 'Histology', 'race', 'Sex', 'tstage', 'nstage', 'brainm', 'liverm', 'surgery', 'Radiation', 'Chemotherapy']
response = 'anxietyanddepression'


# In[113]:


bin_num = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
label = ["8", "16", "32", "64", "128", "256", "512", "1024", "2048", "4096"]


# In[133]:


model = H2OGradientBoostingEstimator(seed=1234)
model.train(x=predictors,y=response,training_frame=data_train,validation_frame=data_test)


# In[137]:


ra_plot = model.residual_analysis_plot(data_test)


# In[136]:


ra_plot = model.varimp_plot()


# In[143]:


aml = H2OAutoML(max_runtime_secs=60, seed=1)
aml.train(x=predictors,y=response, training_frame=data_train)


# In[ ]:


# Explain leader model & compare with all AutoML models
exa = aml.explain(test)


# In[ ]:


# Explain a single H2O model (e.g. leader model from AutoML)
exm = aml.leader.explain(test)

# Explain a generic list of models
# use h2o.explain as follows:
# exl = h2o.explain(model_list, test)


# In[144]:


va_plot = aml.varimp_heatmap()


# In[149]:


mc_plot = aml.model_correlation_heatmap(data_train)


# In[119]:


shap_plot = model.shap_summary_plot(data_train)


# In[120]:


learning_curve_plot = model.learning_curve_plot()


# In[121]:


shap_plot = model.shap_summary_plot(data_test)


# In[126]:


learning_curve_plot = model.learning_curve_plot()


# In[150]:


shapr_plot = model.shap_explain_row_plot(data_test, row_index=0)


# In[151]:


shapr_plot = model.shap_explain_row_plot(data_test, row_index=3444)


# In[157]:


pd_plot = aml.pd_multi_plot(data_test, "Chemotherapy")


# In[158]:


pd_plot = aml.pd_multi_plot(data_test, "age")


# In[159]:


pd_plot = aml.pd_multi_plot(data_train, "age")


# In[160]:


pd_plot = model.pd_plot(data_test, "age")


# In[161]:


pd_plot = model.pd_plot(data_train, "age")


# In[162]:


ice_plot = model.ice_plot(data_test, "age")


# In[ ]:





# In[85]:


# Create h2o dataframes. Make sure to run the "Compute and Compare test metrics" cells to create data_test_X 
# before running these cells##data_test在下方绘制ROC曲线前面有。

htrain = h2o.H2OFrame(pd.concat([data_train_X, data_train['anxietyanddepression']], axis=1))
htest = h2o.H2OFrame(pd.concat([data_test_X, data_test['anxietyanddepression']], axis=1))


# In[86]:


# define cols
x = htrain.columns
y = 'anxietyanddepression'
x.remove(y)


# In[87]:


htrain[y] = htrain[y].asfactor()
htest[y] = htest[y].asfactor()


# In[34]:


# Train Deep Learners for 5 hous##确实需要5个小时
aml_gbm_deep = H2OAutoML(max_runtime_secs = 18000, exclude_algos=['GLM','GBM','DRF','StackedEnsemble'])
aml_gbm_deep.train(x=x, y=y, training_frame=htrain, leaderboard_frame=htest)


# In[35]:


aml_gbm_deep.leaderboard


# In[36]:


# Save best deep learner predictions
h2o_deep_pred = aml_gbm_deep.leader.predict(htest)


# In[37]:


# Save the model
model_path = h2o.save_model(model=aml_gbm_deep.leader, path='E:/R code-LMX/R code for SEER for publishing paper/Data/h2o_deep_learner_may31', force=True)


# In[38]:


model_path


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Compute and compare test metrics

# Transform test data set

# In[81]:


data_test_X = pd.DataFrame(preproc_pipeline.transform(data_test), # it's imperative not to do fit_transfomr again
                           columns=categorical_vars + continuous_vars)


# In[82]:


data_test_X.shape


# In[83]:


data_test_X.head()


# Compute test accuracy score

# In[84]:


def plot_roc_curve(fpr, tpr, auc, model=None):
    if model == None:
        title = None
    elif isinstance(model, str):
        title = model
    else:
        title = model.__class__.__name__
#    title = None if model == None else model.__class__.__name__
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, linewidth=2, label='auc: {}'.format(auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-.01, 1.01, -.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(fontsize=14).get_frame().set_facecolor('white')
    plt.title('{} - ROC Curve'.format(title))


# # Logistic Regression model

# In[35]:


# Import model and retrain
with open('E:/R code-LMX/R code for SEER for publishing paper/Data/lr_clf_final_round.pkl', 'rb') as f:
    lr_clf = pickle.load(f)
lr_clf.fit(data_train_X, data_train['anxietyanddepression'])


# Accuracy scores

# In[36]:


accu_lr = accuracy_score(data_test['anxietyanddepression'], lr_clf.predict(data_test_X))


# In[37]:


round(accu_lr,3)


# In[38]:


pd.crosstab(data_test['anxietyanddepression'], lr_clf.predict(data_test_X))


# ROC and AUC

# In[39]:


pred_proba_lr = lr_clf.predict_proba(data_test_X)


# In[57]:





# In[58]:





# In[40]:


fpr, tpr, _ = roc_curve(data_test['anxietyanddepression'], pred_proba_lr[:, 1])
auc_lr = roc_auc_score(data_test['anxietyanddepression'], pred_proba_lr[:, 1])


# In[41]:


round(auc_lr,3)


# In[42]:


plot_roc_curve(fpr, tpr, round(auc_lr,3), lr_clf)


# Concat prediction_proba for each model to test set, save test set

# In[43]:


data_test['lr_pred_proba'] = pred_proba_lr[:, 1]


# In[44]:


data_test.to_csv('E:/R code-LMX/R code for SEER for publishing paper/Data/test_set_with_predictions-lr.csv'.format(len(data_train)), index=False)


# # XGBc

# In[45]:


# Import model and retrain
with open('E:/R code-LMX/R code for SEER for publishing paper/Data/Xgbc_clf_final_round.pkl', 'rb') as f:
    Xgbc_clf = pickle.load(f)
Xgbc_clf.fit(data_train_X, data_train['anxietyanddepression'])


# In[46]:


accu_Xgbc = accuracy_score(data_test['anxietyanddepression'], Xgbc_clf.predict(data_test_X))


# In[47]:


round(accu_Xgbc,3)


# In[48]:


pd.crosstab(data_test['anxietyanddepression'], Xgbc_clf.predict(data_test_X))


# In[75]:


#pred_proba_Xgbc = Xgbc_clf.predict_proba(data_train_X)


# In[76]:


#fpr, tpr, _ = roc_curve(data_train['anxietyanddepression'], pred_proba_Xgbc[:, 1])
#auc_Xgbc = roc_auc_score(data_train['anxietyanddepression'], pred_proba_Xgbc[:, 1])


# In[49]:


pred_proba_Xgbc = Xgbc_clf.predict_proba(data_test_X)


# In[50]:


fpr, tpr, _ = roc_curve(data_test['anxietyanddepression'], pred_proba_Xgbc[:, 1])
auc_Xgbc = roc_auc_score(data_test['anxietyanddepression'], pred_proba_Xgbc[:, 1])


# In[51]:


round(auc_Xgbc,3)


# In[52]:


plot_roc_curve(fpr, tpr, round(auc_Xgbc,3), Xgbc_clf)


# In[92]:


data_test['lr_pred_proba'] = pred_proba_Xgbc[:, 1]


# In[93]:


data_test.to_csv('E:/R code-LMX/R code for SEER for publishing paper/Data/test_set_with_predictions-Xgbc.csv'.format(len(data_train)), index=False)


# In[ ]:





# # Random forests model

# In[570]:


# Import model and retrain
with open('E:/R code-LMX/R code for SEER for publishing paper/Data/rf_clf_final_round.pkl', 'rb') as f:
    rf_clf = pickle.load(f)
rf_clf.fit(data_train_X, data_train['anxietyanddepression'])


# Accuracy scores

# In[571]:


accu_rf = accuracy_score(data_test['anxietyanddepression'], rf_clf.predict(data_test_X))


# In[572]:


round(accu_rf,3)


# In[573]:


pd.crosstab(data_test['anxietyanddepression'], rf_clf.predict(data_test_X))


# ROC and AUC

# In[574]:


pred_proba_rf = rf_clf.predict_proba(data_test_X)


# In[575]:


fpr, tpr, _ = roc_curve(data_test['anxietyanddepression'], pred_proba_rf[:, 1])
auc_rf = roc_auc_score(data_test['anxietyanddepression'], pred_proba_rf[:, 1])


# In[576]:


round(auc_rf,3)


# In[577]:


plot_roc_curve(fpr, tpr, round(auc_rf,3), rf_clf)


# In[578]:


data_test['lr_pred_proba'] = pred_proba_rf[:, 1]


# In[579]:


data_test.to_csv('E:/R code-LMX/R code for SEER for publishing paper/Data/test_set_with_predictions-Random forests model.csv'.format(len(data_train)), index=False)


# # Gradient boosting machine model

# In[371]:


# Import model and retrain
with open('E:/R code-LMX/R code for SEER for publishing paper/Data/gbm_clf_final_round.pkl', 'rb') as f:
    gbm_clf = pickle.load(f)
gbm_clf.fit(data_train_X, data_train['anxietyanddepression'])


# Accuracy scores

# In[372]:


accu_gbm = accuracy_score(data_test['anxietyanddepression'], gbm_clf.predict(data_test_X))


# In[373]:


round(accu_gbm,3)


# In[374]:


pd.crosstab(data_test['anxietyanddepression'], gbm_clf.predict(data_test_X))


# ROC and AUC

# In[375]:


pred_proba_gbm = gbm_clf.predict_proba(data_test_X)


# In[376]:


fpr, tpr, _ = roc_curve(data_test['anxietyanddepression'], pred_proba_gbm[:, 1])
auc_gbm = roc_auc_score(data_test['anxietyanddepression'], pred_proba_gbm[:, 1])


# In[131]:


pred_proba_gbm = gbm_clf.predict_proba(data_train_X)


# In[132]:


fpr, tpr, _ = roc_curve(data_train['anxietyanddepression'], pred_proba_gbm[:, 1])
auc_gbm = roc_auc_score(data_train['anxietyanddepression'], pred_proba_gbm[:, 1])


# In[377]:


round(auc_gbm,3)


# In[378]:


plot_roc_curve(fpr, tpr, round(auc_gbm,3), gbm_clf)


# In[379]:


data_test['lr_pred_proba'] = pred_proba_gbm[:, 1]


# In[380]:


data_test.to_csv('E:/R code-LMX/R code for SEER for publishing paper/Data/test_set_with_predictions-Gradient boosting machine model.csv'.format(len(data_train)), index=False)


# # Support vector machine model

# In[537]:


# Import model and retrain
with open('E:/R code-LMX/R code for SEER for publishing paper/Data/svc_clf_final_round.pkl', 'rb') as f:
    svc_clf = pickle.load(f)
svc_clf.fit(data_train_X, data_train['anxietyanddepression'])


# In[538]:


accu_svc = accuracy_score(data_test['anxietyanddepression'], svc_clf.predict(data_test_X))


# In[539]:


round(accu_svc,3)


# In[540]:


pd.crosstab(data_test['anxietyanddepression'], svc_clf.predict(data_test_X))


# In[541]:


pred_proba_svc = svc_clf.predict_proba(data_test_X) 


# In[542]:


svc_clf.predict


# In[543]:


fpr, tpr, _ = roc_curve(data_test['anxietyanddepression'], pred_proba_svc[:, 1])
auc_svc = roc_auc_score(data_test['anxietyanddepression'], pred_proba_svc[:, 1])


# In[544]:


round(auc_svc,3)


# In[545]:


plot_roc_curve(fpr, tpr, round(auc_svc,3), svc_clf)


# In[546]:


data_test['lr_pred_proba'] = pred_proba_svc[:, 1]


# In[547]:


data_test.to_csv('E:/R code-LMX/R code for SEER for publishing paper/Data/test_set_with_predictions-Support vector machine model.csv'.format(len(data_train)), index=False)


# # The ensemble model

# In[551]:


# Import model and retrain
with open('E:/R code-LMX/R code for SEER for publishing paper/Data/ensemble_clf_final_round.pkl', 'rb') as f:
    ensemble_clf = pickle.load(f)
ensemble_clf.fit(data_train_X, data_train['anxietyanddepression'])


# In[552]:


accu_ensemble = accuracy_score(data_test['anxietyanddepression'], ensemble_clf.predict(data_test_X))


# In[553]:


round(accu_ensemble,3)


# In[554]:


pd.crosstab(data_test['anxietyanddepression'], ensemble_clf.predict(data_test_X))


# ROC and AUC

# In[555]:


ensemble_clf.predict(data_test_X)


# In[556]:


ensemble_clf.predict_proba(data_test_X)


# In[557]:


pred_proba_ensemble = ensemble_clf.predict_proba(data_test_X)


# In[558]:


fpr, tpr, _ = roc_curve(data_test['anxietyanddepression'], pred_proba_ensemble[:, 1])
auc_ensemble = roc_auc_score(data_test['anxietyanddepression'], pred_proba_ensemble[:, 1])


# In[559]:


round(auc_ensemble,3)


# In[560]:


plot_roc_curve(fpr, tpr, round(auc_ensemble,3), ensemble_clf)


# In[562]:


data_test['lr_pred_proba'] = pred_proba_ensemble[:, 1]


# In[563]:


data_test.to_csv('E:/R code-LMX/R code for SEER for publishing paper/Data/test_set_with_predictions-The ensemble model.csv'.format(len(data_train)), index=False)


# In[ ]:





# # H2O Deep Learner

# In[39]:


model_path


# In[40]:


# Use manual path if model_path is not defined
h2o_deep_learner = h2o.load_model('E:\\h2o_deep_learner_may31\\DeepLearning_grid_1_AutoML_2_20220504_122615_model_210')
h2o_deep_learner.train(x=x, y=y, training_frame=htrain)


# In[129]:


# Get predictions
h2o_deep_pred = h2o_deep_learner.predict(htest)


# In[88]:


# Convert to pandas df
h2o_deep_pred = h2o_deep_pred['p1'].as_data_frame()


# In[89]:


h2o_deep_pred


# In[90]:


accu_h2o_deep = accuracy_score(data_test['anxietyanddepression'], round(h2o_deep_pred))
accu_h2o_deep


# In[91]:


##下面没有计算出来，有问题
pd.crosstab(data_test['anxietyanddepression'], round(h2o_deep_pred))


# In[92]:


fpr, tpr, _ = roc_curve(data_test['anxietyanddepression'], h2o_deep_pred)
auc_h2o_deep = roc_auc_score(data_test['anxietyanddepression'], h2o_deep_pred)


# In[93]:


auc_h2o_deep


# In[94]:


plot_roc_curve(fpr, tpr, auc_h2o_deep, h2o_deep_learner)


# In[53]:


data_test['lr_pred_proba'] = h2o_deep_pred


# In[54]:


data_test.to_csv('E:/test_set_with_predictions-h2o_deep_pred.csv'.format(len(data_train)), index=False)


# In[ ]:





# Class breakdown per model
# 下面的代码中
# [df.g == 0]，df.后面为y变量，此处为anxietyanddepression

# In[88]:


def plot_class_breakdown_hist(df, var, var_name, plot_title, xlog=False, ylog=False, **histkwargs):
    df[var][df.anxietyanddepression == 0].hist(alpha=.5, label='Negative', color = "green", **histkwargs)
    df[var][df.anxietyanddepression == 1].hist(alpha=.5, label='Positive', color = "red", **histkwargs)
    plt.xlabel(var_name)
    plt.title(plot_title)
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    plt.ylim(ymax=35, ymin=0)
    plt.legend()
    plt.savefig(var_name + ' Class Breakdown.png');


# In[89]:


plot_class_breakdown_hist(data_test, 'lr_pred_proba', var_name='Logistic Regression Risk', 
                          plot_title='Logistic Regression Class Breakdown', bins=100)


# In[ ]:





# In[ ]:





# In[ ]:




