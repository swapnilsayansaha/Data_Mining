#!/usr/bin/env python
# coding: utf-8

# In[21]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD, NMF, randomized_svd
from sklearn.metrics import auc, roc_curve, plot_roc_curve, plot_confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from scipy import spatial
from gensim.scripts.glove2word2vec import glove2word2vec 
from gensim.models import KeyedVectors
from tempfile import mkdtemp
from shutil import rmtree
from joblib import Memory
from matplotlib import pyplot as plt
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
import nltk
import string
from string import punctuation
import os
import pandas as pd
import umap 
import umap.plot
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
np.random.seed(42)
random.seed(42)


# # Question 1

# In[2]:


newsgroups_train = fetch_20newsgroups(subset='train')
u, inv = np.unique(newsgroups_train.target, return_inverse=True)
plt.bar(u, np.bincount(inv), width=0.7)
locs, labels = plt.xticks()  
plt.grid(linestyle=':')
plt.xticks(np.arange(20), np.array(newsgroups_train.target_names),rotation=90)
plt.ylabel('No. of documents')
plt.xlabel('Category')
plt.savefig('Q1.png',dpi=300,bbox_inches='tight')
plt.show()


# # Question 2

# In[3]:


categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
train_dataset = fetch_20newsgroups(subset = 'train', categories = categories,shuffle = True, random_state = 42)
test_dataset = fetch_20newsgroups(subset = 'test', categories = categories,shuffle = True, random_state = 42)


# In[4]:


lemmatizer = WordNetLemmatizer()
vectorizer = CountVectorizer(stop_words='english',min_df=3)
tfidf_transformer = TfidfTransformer()

def penn2morphy(penntag): #reference: discussion notebook
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' 

def lemmatizer_func(sentence): 
    lemmatized_sen = []
    lemma_list = [lemmatizer.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(word_tokenize(sentence))]
    for lemma in lemma_list:
        if (not any(char in lemma for char in punctuation) and not any(char.isdigit() for char in lemma)):
            lemmatized_sen.append(lemma.lower())
    return ' '.join(lemmatized_sen)   


# In[5]:


train_data_proc = []
test_data_proc = []
for i in range(len(train_dataset.data)):
    train_data_proc.append(lemmatizer_func(train_dataset.data[i]))
for i in range(len(test_dataset.data)):
    test_data_proc.append(lemmatizer_func(test_dataset.data[i]))
    
train_data_feat_vec = vectorizer.fit_transform(train_data_proc)
test_data_feat_vec = vectorizer.transform(test_data_proc)
train_data_feat = tfidf_transformer.fit_transform(train_data_feat_vec)
test_data_feat = tfidf_transformer.transform(test_data_feat_vec)


# In[6]:


print(train_data_feat.shape)
print(test_data_feat.shape)


# # Question 3

# In[7]:


svd = TruncatedSVD(n_components=50, random_state=42)
nmf = NMF(n_components=50, init='random', random_state=42)

train_data_LSI = svd.fit_transform(train_data_feat)
print('LSI Train Data Shape:', train_data_LSI.shape)
train_data_NMF = nmf.fit_transform(train_data_feat)
print('NMF Train Data Shape:', train_data_NMF.shape)
U_tr,S_tr,V_tr = randomized_svd(train_data_feat,n_components=50,random_state=42)
print('LSI (train) error:',np.sum(np.array(train_data_feat - (U_tr.dot(np.diag(S_tr)).dot(V_tr)))**2))
print('NMF (train) error:',np.sum(np.array(train_data_feat - train_data_NMF.dot(nmf.components_))**2))

test_data_LSI = svd.transform(test_data_feat)
print('LSI Test Data Shape:', test_data_LSI.shape)
test_data_NMF = nmf.transform(test_data_feat)
print('NMF Test Data Shape:', test_data_NMF.shape)
U_te,S_te,V_te = randomized_svd(test_data_feat,n_components=50,random_state=42)
print('LSI (test) error:',np.sum(np.array(test_data_feat - (U_te.dot(np.diag(S_te)).dot(V_te)))**2))
print('NMF (test) error:',np.sum(np.array(test_data_feat - test_data_NMF.dot(nmf.components_))**2))


# # Question 4

# In[8]:


y_train = []
y_test = []
for label in train_dataset.target:
    if label < 4:
        y_train.append(0)
    else:
        y_train.append(1)
for label in test_dataset.target:
    if label < 4:
        y_test.append(0)
    else:
        y_test.append(1)


# In[9]:


clf_hard = svm.SVC(kernel='linear',C=1000,random_state=42)
clf_soft = svm.SVC(kernel='linear',C=0.0001,random_state=42)
pred_hard = clf_hard.fit(train_data_LSI, y_train).predict(test_data_LSI)
pred_soft = clf_soft.fit(train_data_LSI, y_train).predict(test_data_LSI)


# In[56]:


fig, ax = plt.subplots()
fpr, tpr, _ = roc_curve(y_test, clf_hard.decision_function(test_data_LSI))
plot_roc_curve(clf_hard, test_data_LSI, y_test, ax=ax, color='g',label="AUC: "+str(auc(fpr,tpr))) 
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.5)
plt.title('ROC characteristics for hard SVM')
plt.savefig('Q41.png',dpi=300,bbox_inches='tight')
plt.show()


fig, ax = plt.subplots()
fpr, tpr, _ = roc_curve(y_test, clf_soft.decision_function(test_data_LSI))
plot_roc_curve(clf_soft, test_data_LSI, y_test, ax=ax, color='b',label="AUC: "+str(auc(fpr,tpr))) 
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.5)
plt.title('ROC characteristics for soft SVM')
plt.savefig('Q42.png',dpi=300,bbox_inches='tight')
plt.show()


# In[28]:


plot_confusion_matrix(clf_hard, test_data_LSI, y_test,display_labels=['Comp. Tech.','Rec. Act.'])
plt.title('Hard SVM')
plt.savefig('Q43.png',dpi=300,bbox_inches='tight')
plt.show()
plot_confusion_matrix(clf_soft, test_data_LSI, y_test,display_labels=['Comp. Tech.','Rec. Act.']) 
plt.title('Soft SVM')
plt.savefig('Q44.png',dpi=300,bbox_inches='tight')
plt.show()


# In[29]:


print("Accuracy (hard SVM):", accuracy_score(y_test,pred_hard))
print("Recall (hard SVM):", recall_score(y_test,pred_hard))
print("Precision (hard SVM):", precision_score(y_test,pred_hard))
print("F1-Score (hard SVM):", f1_score(y_test,pred_hard))
print("Accuracy (soft SVM):", accuracy_score(y_test,pred_soft))
print("Recall (soft SVM):", recall_score(y_test,pred_soft))
print("Precision (soft SVM):", precision_score(y_test,pred_soft))
print("F1-Score (soft SVM):", f1_score(y_test,pred_soft))


# In[30]:


clf_cv = svm.SVC(random_state=42)
param_grid = {'C': [0.001,0.01,0.1,1,10,100,200,400,600,800,1000],  
              'kernel': ['linear']}
grid = GridSearchCV(clf_cv,param_grid,cv=5,scoring='accuracy')
grid.fit(train_data_LSI,y_train)
pred_cv = grid.best_estimator_.predict(test_data_LSI)


# In[31]:


print('Best Value of gamma:',grid.best_params_['C']) 
for l, n in zip(param_grid['C'],grid.cv_results_['mean_test_score']):
    print(f'Gamma: {l}\t',f'Avg. Validation Accuracy: {n}')


# In[55]:


fig, ax = plt.subplots()
fpr, tpr, _ = roc_curve(y_test, grid.best_estimator_.decision_function(test_data_LSI))
plot_roc_curve(grid.best_estimator_, test_data_LSI, y_test, ax=ax, color='r',label="AUC: "+str(auc(fpr,tpr))) 
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.5)
plt.title('ROC characteristics for Best SVM')
plt.savefig('Q45.png',dpi=300,bbox_inches='tight')
plt.show()


# In[33]:


plot_confusion_matrix(grid.best_estimator_, test_data_LSI, y_test,display_labels=['Comp. Tech.','Rec. Act.'])
plt.title('Best SVM')
plt.savefig('Q46.png',dpi=300,bbox_inches='tight')
plt.show()


# In[34]:


print("Accuracy (best SVM):", accuracy_score(y_test,pred_cv))
print("Recall (best SVM):", recall_score(y_test,pred_cv))
print("Precision (best SVM):", precision_score(y_test,pred_cv))
print("F1-Score (best SVM):", f1_score(y_test,pred_cv))


# ## Question 5

# In[35]:


clf_lr_wor = LogisticRegression(C=1000000,random_state=42,max_iter=100000)
pred_lr_wor = clf_lr_wor.fit(train_data_LSI,y_train).predict(test_data_LSI)


# In[54]:


fig, ax = plt.subplots()
fpr, tpr, _ = roc_curve(y_test, clf_lr_wor.decision_function(test_data_LSI))
plot_roc_curve(clf_lr_wor, test_data_LSI, y_test, ax=ax, color='b',label="AUC: "+str(auc(fpr,tpr))) 
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.5)
plt.title('ROC characteristics for Logistic Classifier (w/o regularization)')
plt.savefig('Q51.png',dpi=300,bbox_inches='tight')
plt.show()


# In[37]:


plot_confusion_matrix(clf_lr_wor, test_data_LSI, y_test,display_labels=['Comp. Tech.','Rec. Act.'])
plt.title('Logistic Classifier (w/o regularization)')
plt.savefig('Q52.png',dpi=300,bbox_inches='tight')
plt.show()


# In[38]:


print("Accuracy (Logistic Classifier - w/o regularization):", accuracy_score(y_test,pred_lr_wor))
print("Recall (Logistic Classifier - w/o regularization):", recall_score(y_test,pred_lr_wor))
print("Precision (Logistic Classifier - w/o regularization):", precision_score(y_test,pred_lr_wor))
print("F1-Score (Logistic Classifier - w/o regularization):", f1_score(y_test,pred_lr_wor))


# In[39]:


clf_lr_l1 = LogisticRegression(penalty='l1',random_state=42,solver='liblinear',max_iter=100000)
param_grid = {'C': [0.001,0.01,0.1,1,10,100,200,400,600,800,1000]}
grid_l1 = GridSearchCV(clf_lr_l1,param_grid,cv=5,scoring='accuracy')
grid_l1.fit(train_data_LSI,y_train)
pred_cv_lr_l1 = grid_l1.best_estimator_.predict(test_data_LSI)

clf_lr_l2 = LogisticRegression(penalty='l2',solver='liblinear',random_state=42)
grid_l2 = GridSearchCV(clf_lr_l2,param_grid,cv=5,scoring='accuracy')
grid_l2.fit(train_data_LSI,y_train)
pred_cv_lr_l2 = grid_l2.best_estimator_.predict(test_data_LSI)


# In[40]:


print('Best Value of L1 Regularization Parameter:',grid_l1.best_params_['C']) 
for l, n in zip(param_grid['C'],grid_l1.cv_results_['mean_test_score']):
    print(f'L1 Reg. Param.: {l}\t',f'Avg. Validation Accuracy: {n}')
    
print('Best Value of L2 Regularization Parameter:',grid_l2.best_params_['C']) 
for l, n in zip(param_grid['C'],grid_l2.cv_results_['mean_test_score']):
    print(f'L2 Reg. Param.: {l}\t',f'Avg. Validation Accuracy: {n}')


# In[41]:


print("Accuracy (best logistic classifer with L1 regularization):", accuracy_score(y_test,pred_cv_lr_l1 ))
print("Recall (best logistic classifer with L1 regularization):", recall_score(y_test,pred_cv_lr_l1 ))
print("Precision (best logistic classifer with L1 regularization):", precision_score(y_test,pred_cv_lr_l1 ))
print("F1-Score (best logistic classifer with L1 regularization):", f1_score(y_test,pred_cv_lr_l1 ))
print("Accuracy (best logistic classifer with L2 regularization):", accuracy_score(y_test,pred_cv_lr_l2 ))
print("Recall (best logistic classifer with L2 regularization):", recall_score(y_test,pred_cv_lr_l2 ))
print("Precision (best logistic classifer with L2 regularization):", precision_score(y_test,pred_cv_lr_l2 ))
print("F1-Score (best logistic classifer with L2 regularization):", f1_score(y_test,pred_cv_lr_l2))


# In[42]:


C_list = [0.001,0.01,0.1,1,10,100,200,400,600,800,1000]
accu_coeff_l1 = []
mean_coeff_l1 = []
accu_coeff_l2 = []
mean_coeff_l2 = []
for j in C_list:
    clf_lr_l1_coeff = LogisticRegression(C=j,penalty='l1',random_state=42,solver='liblinear',max_iter=100000) 
    pred_lr_l1_coeff = clf_lr_l1_coeff.fit(train_data_LSI,y_train).predict(test_data_LSI)
    accu_coeff_l1.append(accuracy_score(y_test,pred_lr_l1_coeff))
    mean_coeff_l1.append(np.mean(clf_lr_l1_coeff.coef_))
    clf_lr_l2_coeff = LogisticRegression(C=j,penalty='l2',random_state=42,solver='liblinear') 
    pred_lr_l2_coeff = clf_lr_l2_coeff.fit(train_data_LSI,y_train).predict(test_data_LSI)
    accu_coeff_l2.append(accuracy_score(y_test,pred_lr_l2_coeff))
    mean_coeff_l2.append(np.mean(clf_lr_l2_coeff.coef_))


# In[43]:


fig, ax = plt.subplots()
plt.title('Effect of reg. strength on logistic classifier coefficients')
plt.plot(C_list,mean_coeff_l1,label='L1')
plt.plot(C_list,mean_coeff_l2,label='L2')
plt.xlabel('Value of C (inv. prop. to reg. strength)')
plt.ylabel('Avg. of logistic coefficients')
plt.legend()
plt.savefig('Q53.png',dpi=300,bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
plt.title('Effect of regularization strength on logistic classifier accuracy')
plt.plot(C_list[0:4],accu_coeff_l1[0:4],label='L1')
plt.plot(C_list[0:4],accu_coeff_l2[0:4],label='L2')
plt.xlabel('Value of C (inv. prop. to reg. strength)')
plt.ylabel('Mean test accuracy')
plt.legend()
plt.savefig('Q54.png',dpi=300,bbox_inches='tight')
plt.show()


# In[44]:


print(np.mean(grid_l1.best_estimator_.coef_))
print(np.mean(grid_l2.best_estimator_.coef_))


# ## Question 6

# In[45]:


clf_NB = GaussianNB()
pred_NB = clf_NB.fit(train_data_LSI, y_train).predict(test_data_LSI)


# In[48]:


fig, ax = plt.subplots()
plot_roc_curve(clf_NB, test_data_LSI, y_test, ax=ax, color='b') 
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.5)
plt.title('ROC characteristics for Gaussian Naive Bayes classifer')
plt.savefig('Q61.png',dpi=300,bbox_inches='tight')
plt.show()


# In[30]:


plot_confusion_matrix(clf_NB, test_data_LSI, y_test,display_labels=['Comp. Tech.','Rec. Act.'])
plt.title('Gaussian Naive Bayes')
plt.savefig('Q62.png',dpi=300,bbox_inches='tight')
plt.show()


# In[31]:


print("Accuracy (Gaussian Naive Bayes):", accuracy_score(y_test,pred_NB))
print("Recall (Gaussian Naive Bayes):", recall_score(y_test,pred_NB))
print("Precision (Gaussian Naive Bayes):", precision_score(y_test,pred_NB))
print("F1-Score (Gaussian Naive Bayes):", f1_score(y_test,pred_NB))


# ## Question 7

# In[32]:


def lemmatized(sentence):
    lemmatized_sen = []
    lemma_list = [lemmatizer.lemmatize(word.lower(), pos=penn2morphy(tag)) 
                  for word, tag in pos_tag(word_tokenize(sentence))]
    for lemma in lemma_list:
        if (not any(char in lemma for char in punctuation) and not any(char.isdigit() for char in lemma)):
            lemmatized_sen.append(lemma.lower())
    return lemmatized_sen

def non_lemmatized(sentence):
    non_lemmatized_sen = []
    lemma_list = word_tokenize(sentence)
    for lemma in lemma_list:
        if (not any(char in lemma for char in punctuation) and not any(char.isdigit() for char in lemma)):
            non_lemmatized_sen.append(lemma.lower())  
    return non_lemmatized_sen


# In[33]:


cachedir = mkdtemp()
memory = Memory(location=cachedir, verbose=10)

pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', None),
    ('clf', None),
],
memory=memory
)
param_grid = [
    {
        'vect__min_df': (3,5),
        'vect__analyzer': (lemmatized,non_lemmatized),
        'reduce_dim': (TruncatedSVD(n_components=50, random_state=42), NMF(n_components=50, init='random', random_state=42)), 
        'clf': (svm.SVC(kernel='linear',C=400,random_state=42),
                GaussianNB(),
                LogisticRegression(penalty='l1',C=10,random_state=42,solver='liblinear',max_iter=100000),
                LogisticRegression(C=400,penalty='l2',random_state=42,solver='liblinear')),  
    }
]


# In[34]:


train_dataset_nhf = fetch_20newsgroups(subset = 'train', categories = categories,shuffle = True, random_state = 42,remove=('headers','footers'))
test_dataset_nhf = fetch_20newsgroups(subset = 'test', categories = categories,shuffle = True, random_state = 42,remove=('headers','footers'))
y_train_nhf = []
y_test_nhf = []
for label in train_dataset_nhf.target:
    if label < 4:
        y_train_nhf.append(0)
    else:
        y_train_nhf.append(1)
for label in test_dataset_nhf.target:
    if label < 4:
        y_test_nhf.append(0)
    else:
        y_test_nhf.append(1)


# In[35]:


grid_all_nhf = GridSearchCV(pipeline,cv=5,param_grid=param_grid,scoring='accuracy')
grid_all_nhf.fit(train_dataset_nhf.data, y_train_nhf)


# In[36]:


print(grid_all_nhf.best_estimator_)


# In[37]:


vectorizer_nhf = CountVectorizer(stop_words='english',min_df=5)
tfidf_transformer_nhf = TfidfTransformer()

train_data_nhf_proc = []
test_data_nhf_proc = []
for i in range(len(train_dataset_nhf.data)):
    train_data_nhf_proc.append(lemmatizer_func(train_dataset_nhf.data[i]))
for i in range(len(test_dataset_nhf.data)):
    test_data_nhf_proc.append(lemmatizer_func(test_dataset_nhf.data[i]))
    
train_data_feat_nhf_vec = vectorizer_nhf.fit_transform(train_data_nhf_proc)
test_data_feat_nhf_vec = vectorizer_nhf.transform(test_data_nhf_proc)
train_data_feat_nhf = tfidf_transformer_nhf.fit_transform(train_data_feat_nhf_vec)
test_data_feat_nhf = tfidf_transformer_nhf.transform(test_data_feat_nhf_vec)
svd_nhf = TruncatedSVD(n_components=50, random_state=42)
train_data_LSI_nhf = svd_nhf.fit_transform(train_data_feat_nhf)
test_data_LSI_nhf = svd_nhf.transform(test_data_feat_nhf)


# In[38]:


clf_best_nhf = LogisticRegression(penalty='l2',C=400,solver='liblinear',random_state=42)
pred_best_nhf = clf_best_nhf.fit(train_data_LSI_nhf,y_train_nhf).predict(test_data_LSI_nhf)
print("Test accuracy of best classifier w/o heading and footer in data:", accuracy_score(y_test_nhf,pred_best_nhf))


# In[39]:


grid_all_whf = GridSearchCV(pipeline,cv=5,param_grid=param_grid,scoring='accuracy')
grid_all_whf.fit(train_dataset.data, y_train)


# In[40]:


print(grid_all_whf.best_estimator_)


# In[41]:


clf_best_whf = svm.SVC(kernel='linear',C=400,random_state=42)
pred_best_whf = clf_best_whf.fit(train_data_LSI, y_train).predict(test_data_LSI)
print("Test accuracy of best classifier with heading and footer in data:", accuracy_score(y_test,pred_best_whf))


# ## Question 8 (Part C)

# In[42]:


#Place the glove.6B.300d.txt file in a folder named glove in the project directory
embeddings_dict = {}
dimension_of_glove = 300
with open("glove/glove.6B.300d.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector


# In[43]:


print(np.linalg.norm(embeddings_dict['queen']-embeddings_dict['king']-embeddings_dict['wife']+embeddings_dict['husband']))
print(np.linalg.norm(embeddings_dict['queen']-embeddings_dict['king']))
print(np.linalg.norm(embeddings_dict['wife']-embeddings_dict['husband']))


# In[44]:


root_folder='.'
glove_folder_name='glove'
glove_filename='glove.6B.300d.txt'
glove_path = os.path.abspath(os.path.join(root_folder, glove_folder_name, glove_filename))
word2vec_output_file = glove_filename+'.word2vec'
glove2word2vec(glove_path, word2vec_output_file)
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)


# In[45]:


result = model.similar_by_word("king")
print("king is similar to {}: {:.4f}".format(*result[0]))
result = model.similar_by_word("wife")
print("wife is similar to {}: {:.4f}".format(*result[0]))


# In[46]:


def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

print(find_closest_embeddings(embeddings_dict["queen"] - embeddings_dict["king"] + embeddings_dict["husband"])[:5])


# ## Question 9

# In[47]:


def punc_num_remover(sentence):
    non_lemmatized_sen = []
    lemma_list = word_tokenize(sentence)
    for lemma in lemma_list:
        if (not any(char in lemma for char in punctuation) and not any(char.isdigit() for char in lemma)):
            non_lemmatized_sen.append(lemma.lower())  
    return ' '.join(non_lemmatized_sen)

X_train = []
X_test = []
for i in range(len(train_dataset.data)):
    X_train.append(punc_num_remover(train_dataset.data[i]))
for i in range(len(test_dataset.data)):
    X_test.append(punc_num_remover(test_dataset.data[i]))


# In[48]:


#Reference: 
#https://edumunozsala.github.io/BlogEms/jupyter/nlp/classification/embeddings/python/2020/08/15/Intro_NLP_WordEmbeddings_Classification.html
class Word2VecVectorizer:
    def __init__(self, model):
        print("Loading in word vectors...")
        self.word_vectors = model
        print("Finished loading in word vectors")
    
    def fit(self, data):
        pass
    
    def transform(self, data):
        v = self.word_vectors.get_vector('king')
        self.D = v.shape[0]

        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        for sentence in data:
            tokens = sentence.split()
            vecs = []
            m = 0
            for word in tokens:
                try:
                    vec = self.word_vectors.get_vector(word)
                    vecs.append(vec)
                    m += 1
                except KeyError:
                    pass
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n += 1
        print("Number of samples with no words found: %s / %s" % (emptycount, len(data)))
        return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


# In[49]:


vectorizer = Word2VecVectorizer(model)


# In[50]:


Xtrain = vectorizer.fit_transform(X_train)
Ytrain = y_train
Xtest = vectorizer.transform(X_test)
Ytest = y_test
print(Xtrain.shape,Xtest.shape)


# In[51]:


clf_cv_Glove = svm.SVC(random_state=42)
param_grid = {'C': [0.001,0.01,0.1,1,10,100,200,400,600,800,1000],  
              'kernel': ['linear']}
Glove_model = GridSearchCV(clf_cv_Glove,param_grid,cv=5,scoring='accuracy',n_jobs=-1).fit(Xtrain, Ytrain)
y_pred_glove = Glove_model.best_estimator_.predict(Xtest)


# In[52]:


print(Glove_model.best_estimator_)


# In[53]:


print("Accuracy (Best GLoVE classifier):", accuracy_score(Ytest,y_pred_glove))
print("Recall (Best GLoVE classifier):", recall_score(Ytest,y_pred_glove))
print("Precision (Best GLoVE classifier):", precision_score(Ytest,y_pred_glove))
print("F1-Score (Best GLoVE classifier):", f1_score(Ytest,y_pred_glove))


# ## Question 10

# In[54]:


filenames_glove = ['glove.6B.50d.txt','glove.6B.100d.txt','glove.6B.200d.txt','glove.6B.300d.txt']
accu_list_glove = []
for filename in filenames_glove:
    print('Training for: ', filename)
    glove_filename=filename
    glove_path = os.path.abspath(os.path.join(root_folder, glove_folder_name, glove_filename))
    word2vec_output_file = glove_filename+'.word2vec'
    glove2word2vec(glove_path, word2vec_output_file)
    model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    vectorizer = Word2VecVectorizer(model)
    Xtrain = vectorizer.fit_transform(X_train)
    Ytrain = y_train
    Xtest = vectorizer.transform(X_test)
    Ytest = y_test
    clf_cur = svm.SVC(kernel='linear',C=1,random_state=42)
    pred_cur = clf_cur.fit(Xtrain, Ytrain).predict(Xtest)
    accu_list_glove.append(accuracy_score(Ytest,pred_cur))  


# In[55]:


dim_list = [50,100,200,300]
plt.plot(dim_list,accu_list_glove)
plt.title('Accuracy vs. Dimension of GLoVE for Linear SVM (Gamma = 1)')
plt.xlabel('Dimension of pre-trained GLoVE embedding')
plt.ylabel('Test accuracy')
plt.savefig('Q101.png',dpi=300,bbox_inches='tight')
plt.show()


# ## Question 11

# In[56]:


reduced_dim_embedding = umap.UMAP(n_components=2, metric='euclidean').fit(Xtrain)
print(reduced_dim_embedding.embedding_.shape)


# In[57]:


YtrainTextLabel = []
for label in Ytrain:
    if(label==0):
        YtrainTextLabel.append('Computer Technology')
    else:
        YtrainTextLabel.append('Recreational Activity')


# In[58]:


s = np.random.normal(0, 1, [4732,300])
s = s / np.linalg.norm(s)
reduced_dim_s = umap.UMAP(n_components=2, metric='cosine').fit(s)


# In[59]:


f = umap.plot.points(reduced_dim_embedding, labels=np.array(YtrainTextLabel))
plt.title('2D plot for GLoVE features (n = 300) for 2 classes of the training set')
plt.savefig('Q111.png',dpi=300,bbox_inches='tight')
g = umap.plot.points(reduced_dim_s)
plt.title('2D plot normalized random vectors')
plt.savefig('Q112.png',dpi=300,bbox_inches='tight')
plt.show()


# ## Question 12

# In[73]:


categories_mc = ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','misc.forsale','soc.religion.christian']
train_dataset_mc = fetch_20newsgroups(subset = 'train', categories = categories_mc,shuffle = True, random_state = 42)
test_dataset_mc = fetch_20newsgroups(subset = 'test', categories = categories_mc,shuffle = True, random_state = 42)


# In[76]:


vectorizer_mc = CountVectorizer(stop_words='english',min_df=3)
tfidf_transformer_mc = TfidfTransformer()
train_data_proc_mc = []
test_data_proc_mc = []
for i in range(len(train_dataset_mc.data)):
    train_data_proc_mc.append(lemmatizer_func(train_dataset_mc.data[i]))
for i in range(len(test_dataset_mc.data)):
    test_data_proc_mc.append(lemmatizer_func(test_dataset_mc.data[i]))
    
train_data_feat_vec_mc = vectorizer_mc.fit_transform(train_data_proc_mc)
test_data_feat_vec_mc = vectorizer_mc.transform(test_data_proc_mc)
train_data_feat_mc = tfidf_transformer_mc.fit_transform(train_data_feat_vec_mc)
test_data_feat_mc = tfidf_transformer_mc.transform(test_data_feat_vec_mc)
svd_mc = TruncatedSVD(n_components=50, random_state=42)
train_data_LSI_mc = svd_mc.fit_transform(train_data_feat_mc)
test_data_LSI_mc = svd_mc.transform(test_data_feat_mc)


# In[77]:


clf_NB_mc = GaussianNB()
pred_NB_mc = clf_NB_mc.fit(train_data_LSI_mc, train_dataset_mc.target).predict(test_data_LSI_mc)


# In[78]:


plot_confusion_matrix(clf_NB_mc, test_data_LSI_mc, test_dataset_mc.target,display_labels=categories_mc)
plt.xticks(rotation=90)
plt.title('Gaussian Multiclass Naive Bayes')
plt.savefig('Q121.png',dpi=300,bbox_inches='tight')
plt.show()


# In[79]:


print("Accuracy (Multiclass Gaussian Naive Bayes):", accuracy_score(test_dataset_mc.target,pred_NB_mc))
print("Recall (Multiclass Gaussian Naive Bayes):", recall_score(test_dataset_mc.target,pred_NB_mc,average='weighted'))
print("Precision (Multiclass Gaussian Naive Bayes):", precision_score(test_dataset_mc.target,pred_NB_mc,average='weighted'))
print("F1-Score (Multiclass Gaussian Naive Bayes):", f1_score(test_dataset_mc.target,pred_NB_mc,average='weighted'))


# In[80]:


clf_svm_mc = OneVsOneClassifier(svm.SVC(random_state=42))
param_grid = {'estimator__C': [0.001,0.01,0.1,1,10,100,200,400,600,800,1000],  
              'estimator__kernel': ['linear']}
grid_svm_mc = GridSearchCV(clf_svm_mc,param_grid,cv=5,scoring='accuracy')
grid_svm_mc.fit(train_data_LSI_mc, train_dataset_mc.target)
pred_svm_mc = grid_svm_mc.best_estimator_.predict(test_data_LSI_mc)


# In[81]:


print(grid_svm_mc.best_estimator_)


# In[82]:


plot_confusion_matrix(grid_svm_mc.best_estimator_, test_data_LSI_mc, test_dataset_mc.target,display_labels=categories_mc)
plt.xticks(rotation=90)
plt.title('One-vs-one SVM')
plt.savefig('Q122.png',dpi=300,bbox_inches='tight')
plt.show()


# In[83]:


print("Accuracy (One-vs-one SVM):", accuracy_score(test_dataset_mc.target,pred_svm_mc))
print("Recall (One-vs-one SVM):", recall_score(test_dataset_mc.target,pred_svm_mc,average='weighted'))
print("Precision (One-vs-one SVM):", precision_score(test_dataset_mc.target,pred_svm_mc,average='weighted'))
print("F1-Score (One-vs-one SVM):", f1_score(test_dataset_mc.target,pred_svm_mc,average='weighted'))


# In[84]:


clf_svm_mc_oa = OneVsRestClassifier(svm.SVC(random_state=42))
param_grid = {'estimator__C': [0.001,0.01,0.1,1,10,100,200,400,600,800,1000],  
              'estimator__kernel': ['linear']}
grid_svm_mc_oa = GridSearchCV(clf_svm_mc_oa,param_grid,cv=5,scoring='accuracy')
grid_svm_mc_oa.fit(train_data_LSI_mc, train_dataset_mc.target)
pred_svm_mc_oa = grid_svm_mc_oa.best_estimator_.predict(test_data_LSI_mc)


# In[85]:


print(grid_svm_mc_oa.best_estimator_)


# In[86]:


plot_confusion_matrix(grid_svm_mc_oa.best_estimator_, test_data_LSI_mc, test_dataset_mc.target,display_labels=categories_mc)
plt.xticks(rotation=90)
plt.title('One-vs-all SVM')
plt.savefig('Q123.png',dpi=300,bbox_inches='tight')
plt.show()


# In[87]:


print("Accuracy (One-vs-all SVM):", accuracy_score(test_dataset_mc.target,pred_svm_mc_oa))
print("Recall (One-vs-all SVM):", recall_score(test_dataset_mc.target,pred_svm_mc_oa,average='weighted'))
print("Precision (One-vs-all SVM):", precision_score(test_dataset_mc.target,pred_svm_mc_oa,average='weighted'))
print("F1-Score (One-vs-all SVM):", f1_score(test_dataset_mc.target,pred_svm_mc_oa,average='weighted'))


# In[ ]:




