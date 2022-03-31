#!/usr/bin/env python
# coding: utf-8

# ## Training with TPrime 600GeV to 700GeV & all five As  background (tth, thq, VH, VHq, and gg)
# 
# >This is the training of signal and bacground to get the output as a HDF5 file. The output file futher used to test on the TPrime at 600 GeV, 625GeV, 650GeV, 675GeV, 700GeV, 800GeV, 900GeV, 1000GeV, 1100GeV, and 1200GeV as signal.
# <br>
# > All the training is to check on how our model behaves.
# 

# In[1]:


#importing modules
import numpy as np
import pandas as pd
import math
import glob
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes = True)
sns.set_palette(sns.color_palette("muted"))

#import all modules related to Keras
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, activations
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import adam, adadelta, adagrad

#importing all ROOT modules
import ROOT
from itertools import tee, islice
from ROOT import TFile, TCanvas, TPad, TPaveLabel, TPaveText, TTree, TH1F, TF1
from root_numpy import root2array, tree2array, array2tree, array2root
from ROOT import gROOT, AddressOf
from root_numpy import root2array, rec2array

#importing sklearn modules
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.ticker as ticker
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import tree


# # Importing and Reading all the signal datasets 
# Tprime from 600GeV to 700GeV will be used as the signal.

# In[2]:


Tprime_600 = '/eos/user/p/prsaha/for_Shivam/output_TprimeBToTH_Hgg_M-600_LH_TuneCP5_PSweights_13TeV-madgraph_pythia8.root'
Tprime_625 = '/eos/user/p/prsaha/for_Shivam/output_TprimeBToTH_Hgg_M-625_LH_TuneCP5_PSweights_13TeV-madgraph_pythia8.root'
Tprime_650 = '/eos/user/p/prsaha/for_Shivam/output_TprimeBToTH_Hgg_M-650_LH_TuneCP5_PSweights_13TeV-madgraph_pythia8.root'
Tprime_675 = '/eos/user/p/prsaha/for_Shivam/output_TprimeBToTH_Hgg_M-675_LH_TuneCP5_PSweights_13TeV-madgraph_pythia8.root'
Tprime_700 = '/eos/user/p/prsaha/for_Shivam/output_TprimeBToTH_Hgg_M-700_LH_TuneCP5_PSweights_13TeV-madgraph_pythia8.root'


# ### Reading from the trees in the .root files

# In[3]:


treeName_600 = "tagsDumper/trees/Tprime_600_13TeV_THQLeptonicTag"
treeName_625 = "tagsDumper/trees/Tprime_625_13TeV_THQLeptonicTag"
treeName_650 = "tagsDumper/trees/Tprime_650_13TeV_THQLeptonicTag"
treeName_675 = "tagsDumper/trees/Tprime_675_13TeV_THQLeptonicTag"
treeName_700 = "tagsDumper/trees/Tprime_700_13TeV_THQLeptonicTag"


# #### All the input variables from the ROOT file used in the training

# In[4]:


columns = ['dipho_leadPt',  'dipho_mass',  'dipho_leadEta',  'dipho_subleadEta', 'dipho_leadIDMVA',
           'dipho_subleadIDMVA',              'dipho_lead_haspixelseed',   'dipho_sublead_haspixelseed',
           'n_jets',  'n_bjets',  'n_centraljets',  'lepton_charge',   'lepton_leadPt',  'lepton_leadEta',
            'fwdjet1_pt',       'fwdjet1_eta', 'fwdjet1_discr', 'top_mt',   'dr_tHchainfwdjet',  'dr_leptonbjet',
             'dr_leptonfwdjet',      'dr_bjetfwdjet',      'dr_leadphofwdjet',    'dr_subleadphofwdjet',
             'bjet1_pt',  'bjet2_pt',  'bjet3_pt', 'bjet1_eta', 'bjet2_eta',    'bjet3_eta', 'bjet1_discr',
             'bjet2_discr',   'bjet3_discr',     'jet1_pt',     'jet2_pt',  'jet3_pt',   'jet1_eta', 'jet2_eta',
             'jet3_eta',              'jet1_discr',              'jet2_discr',              'jet3_discr']
             


# 
# #### Reading the input variables from the ROOT file used in the training

# In[5]:


signal_Tprime_600 = root2array(Tprime_600, treeName_600, columns)   #Signal TPrime at 600GeV
signal_Tprime_625 = root2array(Tprime_625, treeName_625, columns)   #Signal TPrime at 625GeV
signal_Tprime_650 = root2array(Tprime_650, treeName_650, columns)   #Signal TPrime at 650GeV
signal_Tprime_675 = root2array(Tprime_675, treeName_675, columns)   #Signal TPrime at 675GeV
signal_Tprime_700 = root2array(Tprime_700, treeName_700, columns)   #Signal TPrime at 700GeV


# ##### Converting into the pandas dataframes

# In[6]:


signal_Tprime600= pd.DataFrame(signal_Tprime_600)
signal_Tprime625= pd.DataFrame(signal_Tprime_625)
signal_Tprime650= pd.DataFrame(signal_Tprime_650)
signal_Tprime675= pd.DataFrame(signal_Tprime_675)
signal_Tprime700= pd.DataFrame(signal_Tprime_700)


# In[7]:


# Reading the shapes and size of the data
print('shape of the signal_Tprime at 600GeV is:',signal_Tprime600.shape, '\n')
print('shape of the signal_Tprime at 625GeV is:',signal_Tprime625.shape, '\n')
print('shape of the signal_Tprime at 650GeV is:',signal_Tprime650.shape, '\n')
print('shape of the signal_Tprime at 675GeV is:',signal_Tprime675.shape, '\n')
print('shape of the signal_Tprime at 700GeV is:',signal_Tprime700.shape, '\n')


# # Importing and Reading all the Background datasets 
# ttH, tHq, VH, VHq, ggH will be used as the signal.

# In[8]:


back_1 = '/eos/home-s/sraj/M.Sc._Thesis/data_files/output_TTGG_0Jets_TuneCP5_13TeV_amcatnlo_madspin_pythia8.root'
back_01= '/eos/home-s/sraj/M.Sc._Thesis/data_files/output_ttHJetToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8.root'
back_02 = '/eos/home-s/sraj/M.Sc._Thesis/data_files/output_VBFHToGG_M125_13TeV_amcatnlo_pythia8.root'
back_03 = '/eos/home-s/sraj/M.Sc._Thesis/data_files/output_VHToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8.root'
back_04 = '/eos/home-s/sraj/M.Sc._Thesis/data_files/output_THQ_ctcvcp_HToGG_M125_13TeV-madgraph-pythia8.root'
back_05= '/eos/home-s/sraj/M.Sc._Thesis/data_files/output_GluGluHToGG_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8.root'


# In[9]:


treeName_back_1 ="tagsDumper/trees/ttgg_13TeV_THQLeptonicTag" 
treeName_back_05 = "tagsDumper/trees/ggh_125_13TeV_THQLeptonicTag"
treeName_back_04 = "tagsDumper/trees/thq_125_13TeV_THQLeptonicTag"
treeName_back_03 = "tagsDumper/trees/vh_125_13TeV_THQLeptonicTag"
treeName_back_02 = 'tagsDumper/trees/vbf_125_13TeV_THQLeptonicTag'
treeName_back_01 = "tagsDumper/trees/tth_125_13TeV_THQLeptonicTag"


# In[10]:


columns = ['dipho_leadPt',  'dipho_mass',  'dipho_leadEta',  'dipho_subleadEta', 'dipho_leadIDMVA',
           'dipho_subleadIDMVA',              'dipho_lead_haspixelseed',   'dipho_sublead_haspixelseed',
           'n_jets',  'n_bjets',  'n_centraljets',  'lepton_charge',   'lepton_leadPt',  'lepton_leadEta',
            'fwdjet1_pt',       'fwdjet1_eta', 'fwdjet1_discr', 'top_mt',   'dr_tHchainfwdjet',  'dr_leptonbjet',
             'dr_leptonfwdjet',      'dr_bjetfwdjet',      'dr_leadphofwdjet',    'dr_subleadphofwdjet',
             'bjet1_pt',  'bjet2_pt',  'bjet3_pt', 'bjet1_eta', 'bjet2_eta',    'bjet3_eta', 'bjet1_discr',
             'bjet2_discr',   'bjet3_discr',     'jet1_pt',     'jet2_pt',  'jet3_pt',   'jet1_eta', 'jet2_eta',
             'jet3_eta',              'jet1_discr',              'jet2_discr',              'jet3_discr']
             


# In[11]:


back_ttgg = root2array(back_1, treeName_back_1, columns)     # ttgg background(Not using this)
back_tth = root2array(back_01, treeName_back_01, columns)      
back_vbf = root2array(back_02, treeName_back_02, columns)
back_vh = root2array(back_03, treeName_back_03, columns)
back_thq = root2array(back_04, treeName_back_04, columns)
back_ggh = root2array(back_05, treeName_back_05, columns)


# In[12]:


back_tth = pd.DataFrame(back_tth)          #tth background dataframe 
back_vbf = pd.DataFrame(back_vbf)             #vbf background dataframe 
back_vh = pd.DataFrame(back_vh)             #vh background dataframe 
back_thq = pd.DataFrame(back_thq) #thq background dataframe 
back_ggh = pd.DataFrame(back_ggh) #ggh background dataframe 
back_ttgg = pd.DataFrame(back_ttgg) #ttgg as the background converted into the data frame


# In[13]:


print('Shape of the ttgg backgground', back_ttgg.shape, '\n')
print('Shape of the tth backgground', back_tth.shape, '\n')
print('Shape of the thq backgground', back_thq.shape, '\n')
print('Shape of the ggh backgground', back_ggh.shape, '\n')
print('Shape of the vh backgground', back_vh.shape, '\n')
print('Shape of the vbf backgground', back_vbf.shape, '\n')
# print('Total shape', back_ttgg.shape + back_tth.shape + back_thq.shape + back_ggh.shape+ back_vh.shape+back_vbf.shape )


# In[14]:


signal_Tprime600['dipho_subleadEta'].plot.hist(bins=100, histtype = 'step', edgecolor = 'purple', alpha = 1,  density=True, fill= False,color = 'orange', label = 'data' )
signal_Tprime700['dipho_subleadEta'].plot.hist(bins=100, histtype = 'step', edgecolor = 'green', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )
back_tth['dipho_subleadEta'].plot.hist(bins=100, histtype = 'step', edgecolor = 'red', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )


# In[15]:


signal_Tprime600['dipho_mass'].plot.hist(bins=100, histtype = 'step', edgecolor = 'purple', alpha = 1,  density=True, fill= False,color = 'orange', label = 'data' )
signal_Tprime700['dipho_mass'].plot.hist(bins=100, histtype = 'step', edgecolor = 'green', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )
back_tth['dipho_mass'].plot.hist(bins=100, histtype = 'step', edgecolor = 'red', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )


# In[16]:



signal_Tprime600['dipho_leadPt'].plot.hist(bins=100, histtype = 'step', edgecolor = 'purple', alpha = 1,  density=True, fill= False,color = 'orange', label = 'data' )
signal_Tprime700['dipho_leadPt'].plot.hist(bins=100, histtype = 'step', edgecolor = 'green', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )
back_tth['dipho_leadPt'].plot.hist(bins=100, histtype = 'step', edgecolor = 'red', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )


# In[17]:


signal_Tprime600['dipho_leadEta'].plot.hist(bins=100, histtype = 'step', edgecolor = 'purple', alpha = 1,  density=True, fill= False,color = 'orange', label = 'data' )
signal_Tprime700['dipho_leadEta'].plot.hist(bins=100, histtype = 'step', edgecolor = 'green', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )
back_tth['dipho_leadEta'].plot.hist(bins=100, histtype = 'step', edgecolor = 'red', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )


# In[18]:



signal_Tprime600['dipho_leadIDMVA'].plot.hist(bins=100, histtype = 'step', edgecolor = 'purple', alpha = 1,  density=True, fill= False,color = 'orange', label = 'data' )
signal_Tprime700['dipho_leadIDMVA'].plot.hist(bins=100, histtype = 'step', edgecolor = 'green', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )
back_tth['dipho_leadIDMVA'].plot.hist(bins=100, histtype = 'step', edgecolor = 'red', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )
back_thq['dipho_leadIDMVA'].plot.hist(bins=100, histtype = 'step', edgecolor = 'blue', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )
back_ttgg['dipho_leadIDMVA'].plot.hist(bins=100, histtype = 'step', edgecolor = 'lime', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )


# In[19]:


signal_Tprime600['bjet1_eta'].plot.hist(bins=100, histtype = 'step', edgecolor = 'purple', alpha = 1,  density=True, fill= False,color = 'orange', label = 'data' )
signal_Tprime700['bjet1_eta'].plot.hist(bins=100, histtype = 'step', edgecolor = 'green', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )
back_tth['bjet1_eta'].plot.hist(bins=100, histtype = 'step', edgecolor = 'red', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )
back_thq['bjet1_eta'].plot.hist(bins=100, histtype = 'step', edgecolor = 'blue', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )
back_ttgg['bjet1_eta'].plot.hist(bins=100, histtype = 'step', edgecolor = 'lime', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )


# In[20]:


signal_Tprime600['jet2_eta'].plot.hist(bins=100, histtype = 'step', edgecolor = 'purple', alpha = 1,  density=True, fill= False,color = 'orange', label = 'data' )
signal_Tprime700['jet2_eta'].plot.hist(bins=100, histtype = 'step', edgecolor = 'green', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )
back_tth['jet2_eta'].plot.hist(bins=100, histtype = 'step', edgecolor = 'red', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )
back_thq['jet2_eta'].plot.hist(bins=100, histtype = 'step', edgecolor = 'blue', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )
back_ttgg['jet2_eta'].plot.hist(bins=100, histtype = 'step', edgecolor = 'lime', alpha = 1,  density=True, fill= False,color = 'red', label = 'data' )


# ## Training over the Tprime  600 GeV to 700GeV with all combined background

# **Training of the TPrime from 600 GeV to 700GeV**. first combing all the datafiles into one further doing the training on the model of the DNN.

# In[70]:


signal = pd.concat((signal_Tprime600, signal_Tprime625, signal_Tprime650, signal_Tprime675, signal_Tprime700),axis = 0)
backgr = pd.concat((back_tth, back_ggh, back_thq, back_vbf, back_vh), axis=0)


# In[22]:


print('Total signal shape is:', signal.shape, '\n')
print('Total background shape is:', backgr.shape)


# In[23]:


X = np.concatenate((signal, backgr))
y = np.concatenate((np.ones(signal.shape[0]),
                    np.zeros(backgr.shape[0])))


# In[24]:


X.shape, y.shape


# In[25]:


X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state =5)


# In[26]:


X_train.shape, y_train.shape


# In[27]:


X_test.shape, y_test.shape


# In[28]:


from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import keras
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.utils.vis_utils import plot_model
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import RandomNormal, Constant
from keras import callbacks
from keras.models import load_model


# In[29]:


clf = Sequential()
# clf.add(LSTM(1, return_sequences=True ))
clf.add(BatchNormalization(input_shape = (42,)))
initializer =tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
# clf.add(Dropout(0.3))
clf.add(Dense(1024, activation = 'relu',kernel_regularizer=regularizers.l2(0.001), kernel_initializer = 'random_uniform' ,name = 'dense_1'))
clf.add(BatchNormalization())
clf.add(Dropout(0.3))
clf.add(Dense(512, activation = 'relu',kernel_regularizer=regularizers.l2(0.001), kernel_initializer = 'random_uniform', name = 'dense_2'))
clf.add(Dropout(0.35))
clf.add(Dense(256, activation = 'relu',kernel_regularizer=regularizers.l2(0.001), kernel_initializer = 'random_uniform', name = 'dense_3'))
clf.add(BatchNormalization())
clf.add(Dropout(0.40))
clf.add(Dense(128, activation = 'relu',kernel_regularizer=regularizers.l2(0.001), kernel_initializer = 'random_uniform' ,name = 'dense_4'))
clf.add(BatchNormalization())
clf.add(Dropout(0.5))
clf.add(Dense(64, activation = 'relu',kernel_regularizer=regularizers.l2(0.001), kernel_initializer = 'random_uniform', name = 'dense_5'))
clf.add(BatchNormalization(momentum=0.99,epsilon=0.001,beta_initializer=RandomNormal(mean=0.0, stddev=0.05),gamma_initializer=Constant(value=0.9)))
clf.add(Dense(32, activation = 'relu', name = 'dense_6'))
clf.add(Dropout(0.55))
clf.add(BatchNormalization(momentum=0.99,epsilon=0.001,beta_initializer=RandomNormal(mean=0.0, stddev=0.05),gamma_initializer=Constant(value=0.9)))

# Output
clf.add(Dense(1, activation = 'sigmoid',kernel_regularizer=regularizers.l2(0.001), kernel_initializer = 'random_uniform', name = 'output'))
#compile model

# opt = SGD(lr=0.01, momentum=0.9)
clf.compile(loss = 'binary_crossentropy', 
            optimizer= 'adam',
            metrics=['accuracy'])
print('Summary of the built model...')
print(clf.summary())
# plot_model(clf, to_file='/eos/home-s/sraj/M.Sc._Thesis/Plot/''clf_plot_multiclass___.png', show_shapes=True, show_layer_names=True)


# In[37]:


# simple early stopping
early_stopper = callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="auto")
model_check_point = callbacks.ModelCheckpoint("model(Training with Tprime 600-700GeV with all backgrounds).h5", monitor = 'val_loss', verbose=True, 
                                              save_best_only=True, mode='auto')
# fit model
history = clf.fit(X_train, y_train, validation_split = 0.30, batch_size= 9000, epochs=100, verbose=1, callbacks=[early_stopper, model_check_point])
# evaluate the model


# In[38]:


# Final evaluation of the model for DNN
# Testing Outputs
scores = clf.evaluate(X_train, y_train, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[39]:


# Final evaluation of the model for DNN
# Testing Outputs"$\pm$"
scores = clf.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[40]:


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('model_accuracy_TPrime_ttgg_&tth&thq.png')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
# plt.title('model loss,random_state=5, epoch =100,batch_size =900, verbose=0.25')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
# plt.savefig('loss_TPrime_ttgg_&tth&thq.png')
plt.show()


# In[41]:


from sklearn.metrics import roc_curve, auc

decisions = clf.predict(X_test)

# Compute ROC curve and area under the curve
fpr, tpr, thresholds = roc_curve(y_test, decisions)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))


plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid()
# plt.savefig("ROC_curve.png")
plt.show()


# In[42]:


import matplotlib
plt.figure()                                     # new window
matplotlib.rcParams.update({'font.size':14})     # set all font sizes
tTest = clf.predict(X_test)
# if hasattr(clf, "decision_function"):
#     tTest = clf.decision_function(X_test)        # if available use decision_function
# else:
#     tTest = clf.predict_proba(X_test)[:,1]       # for e.g. MLP need to use predict_proba
tBkg_0 = tTest[y_test==0]
tSig_0 = tTest[y_test==1]
nBins = 20
tMin = np.floor(np.min(tTest))
tMax = np.ceil(np.max(tTest))
bins = np.linspace(tMin, tMax, nBins+1)
# plt.title('Multilayer perceptron')
plt.xlabel(' $DNN$', labelpad=3)
plt.ylabel('$Probability density$', labelpad=40)
n, bins, patches = plt.hist(tSig_0 ,bins=bins, density=True, histtype='step', fill=False, color ='dodgerblue' ,edgecolor = 'blue', hatch = 'XX',label='Tprime_600-700')
n, bins, patches = plt.hist(tBkg_0, bins=bins, density=True, histtype='step', fill=False,color = 'red' ,alpha=0.5, edgecolor = 'green', hatch='++', label = 'All Background')
plt.grid(color = 'b', alpha = 0.5, linestyle = 'dashed')
plt.legend(loc='center')
plt.title(' ')
plt.savefig('output_TPrime600_all_background.png')
plt.show()


# In[29]:


loaded_model = load_model("model(Training with Tprime 600-700GeV with all backgrounds).h5")


# In[30]:


# evaluate loaded model on test data of TPrime_600
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test)


# In[31]:


scores = loaded_model.evaluate(X_test, y_test, verbose=1)
print('It should be compatible with the previous output from the test dataset: \n'"Here the Accuracy ouput is: \n Accuracy: %.2f%%" % (scores[1]*100))


# In[32]:


print(X_test.shape)
print(y_test.shape)


# # Testing on Tprime 800 and 900

# In[33]:


Tprime_800 = '/eos/user/p/prsaha/for_Shivam/output_TprimeBToTH_Hgg_M-800_LH_TuneCP5_PSweights_13TeV-madgraph_pythia8.root'
Tprime_900 = '/eos/user/p/prsaha/for_Shivam/output_TprimeBToTH_Hgg_M-900_LH_TuneCP5_PSweights_13TeV-madgraph_pythia8.root'


# In[34]:


treeName_800 = "tagsDumper/trees/Tprime_800_13TeV_THQLeptonicTag"
treeName_900 = "tagsDumper/trees/Tprime_900_13TeV_THQLeptonicTag"


# In[35]:


signal_Tprime_800 = root2array(Tprime_800, treeName_800, columns)   #Signal TPrime at 600GeV
signal_Tprime_900 = root2array(Tprime_900, treeName_900, columns)   #Signal TPrime at 625GeV


# In[36]:


signal_Tprime800= pd.DataFrame(signal_Tprime_800)
signal_Tprime900= pd.DataFrame(signal_Tprime_900)


# In[38]:


signal_1 = pd.concat((signal_Tprime800, signal_Tprime900 ),axis = 0)
backgr = pd.concat((back_tth, back_ggh, back_thq, back_vbf, back_vh), axis=0)


# In[39]:


X_1 = np.concatenate((signal_1, backgr))
y_1 = np.concatenate((np.ones(signal_1.shape[0]),
                    np.zeros(backgr.shape[0])))


# In[40]:


X_1_train,X_1_test, y_1_train, y_1_test = train_test_split(X_1, y_1, test_size=0.99, random_state =5)


# In[41]:


loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_1_test, y_1_test)


# In[42]:


# Final evaluation of the model for DNN
# Testing Outputs
scores = loaded_model.evaluate(X_1_test, y_1_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[43]:


from sklearn.metrics import roc_curve, auc

decisions = loaded_model.predict(X_1_test)

# Compute ROC curve and area under the curve
fpr, tpr, thresholds = roc_curve(y_1_test, decisions)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid()
# plt.savefig("ROC_curve.png")
plt.show()


# In[45]:


import matplotlib
plt.figure()                                     # new window
matplotlib.rcParams.update({'font.size':14})     # set all font sizes
tTest = loaded_model.predict(X_1_test)
# if hasattr(clf, "decision_function"):
#     tTest = clf.decision_function(X_test)        # if available use decision_function
# else:
#     tTest = clf.predict_proba(X_test)[:,1]       # for e.g. MLP need to use predict_proba
tBkg_1 = tTest[y_1_test==0]
tSig_1 = tTest[y_1_test==1]
nBins = 20
tMin = np.floor(np.min(tTest))
tMax = np.ceil(np.max(tTest))
bins = np.linspace(tMin, tMax, nBins+1)
# plt.title('Multilayer perceptron')
plt.xlabel(' $DNN$', labelpad=3)
plt.ylabel('$Probability density$', labelpad=40)
n, bins, patches = plt.hist(tSig_1, bins=bins, density=True, histtype='step', fill=False, color ='dodgerblue' ,edgecolor = 'blue', hatch = 'XX',label='Tprime_800-900')
n, bins, patches = plt.hist(tBkg_1, bins=bins, density=True, histtype='step', fill=False,color = 'red' ,alpha=0.5, edgecolor = 'green', hatch='++', label = 'All Background')
plt.grid(color = 'b', alpha = 0.5, linestyle = 'dashed')
plt.legend(loc='center')
plt.title(' ')
# plt.savefig('/eos/home-s/sraj/M.Sc._Thesis/Plot_M.Sc._thesis/Plot_with_HDF5_files/''output_TPrime1200_all_background.png')
plt.show()


# # Method 2

# In[46]:


X_2_test = signal_1
y_2_test = np.ones(signal_1.shape[0])


# In[48]:


X_2_test.shape, y_2_test.shape


# In[49]:


loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_2_test, y_2_test)


# In[51]:


# Final evaluation of the model for DNN
# Testing Outputs
scores = loaded_model.evaluate(X_2_test, y_2_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[52]:


from sklearn.metrics import roc_curve, auc

decisions = loaded_model.predict(X_2_test)

# Compute ROC curve and area under the curve
fpr, tpr, thresholds = roc_curve(y_2_test, decisions)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid()
# plt.savefig("ROC_curve.png")
plt.show()


# In[53]:


import matplotlib
plt.figure()                                     # new window
matplotlib.rcParams.update({'font.size':14})     # set all font sizes
tTest = loaded_model.predict(X_2_test)
# if hasattr(clf, "decision_function"):
#     tTest = clf.decision_function(X_test)        # if available use decision_function
# else:
#     tTest = clf.predict_proba(X_test)[:,1]       # for e.g. MLP need to use predict_proba
tBkg_1 = tTest[y_2_test==0]
tSig_1 = tTest[y_2_test==1]
nBins = 20
tMin = np.floor(np.min(tTest))
tMax = np.ceil(np.max(tTest))
bins = np.linspace(tMin, tMax, nBins+1)
# plt.title('Multilayer perceptron')
plt.xlabel(' $DNN$', labelpad=3)
plt.ylabel('$Probability density$', labelpad=40)
n, bins, patches = plt.hist(tSig_1, bins=bins, density=True, histtype='step', fill=False, color ='dodgerblue' ,edgecolor = 'blue', hatch = 'XX',label='Tprime_800-900')
n, bins, patches = plt.hist(tBkg_1, bins=bins, density=True, histtype='step', fill=False,color = 'red' ,alpha=0.5, edgecolor = 'green', hatch='++', label = 'All Background')
plt.grid(color = 'b', alpha = 0.5, linestyle = 'dashed')
plt.legend(loc='center')
plt.title(' ')
# plt.savefig('/eos/home-s/sraj/M.Sc._Thesis/Plot_M.Sc._thesis/Plot_with_HDF5_files/''output_TPrime1200_all_background.png')
plt.show()


# # Testing on Tprime 1000 and 900

# In[56]:


Tprime_1000 = '/eos/user/p/prsaha/for_Shivam/output_TprimeBToTH_Hgg_M-1000_LH_TuneCP5_PSweights_13TeV-madgraph_pythia8.root'
Tprime_1100 = '/eos/user/p/prsaha/for_Shivam/output_TprimeBToTH_Hgg_M-1100_LH_TuneCP5_PSweights_13TeV-madgraph_pythia8.root'
Tprime_1200 = '/eos/user/p/prsaha/for_Shivam/output_TprimeBToTH_Hgg_M-1200_LH_TuneCP5_PSweights_13TeV-madgraph_pythia8.root'


# In[57]:


treeName_1000 = "tagsDumper/trees/Tprime_1000_13TeV_THQLeptonicTag"
treeName_1100 = "tagsDumper/trees/Tprime_1100_13TeV_THQLeptonicTag"
treeName_1200 = "tagsDumper/trees/Tprime_1200_13TeV_THQLeptonicTag"


# In[58]:


signal_Tprime_1000 = root2array(Tprime_1000, treeName_1000, columns)   #Signal TPrime at 600GeV
signal_Tprime_1100 = root2array(Tprime_1100, treeName_1100, columns)   #Signal TPrime at 625GeV
signal_Tprime_1200 = root2array(Tprime_1200, treeName_1200, columns)   #Signal TPrime at 625GeV


# In[59]:


signal_Tprime1000= pd.DataFrame(signal_Tprime_1000)
signal_Tprime1100= pd.DataFrame(signal_Tprime_1100)
signal_Tprime1200= pd.DataFrame(signal_Tprime_1200)


# In[66]:


signal_2 = pd.concat((signal_Tprime1000, signal_Tprime1100, signal_Tprime1200 ),axis = 0)
backgr = pd.concat((back_tth, back_ggh, back_thq, back_vbf, back_vh), axis=0)


# In[63]:


X_2 = np.concatenate((signal_2, backgr))
y_2 = np.concatenate((np.ones(signal_1.shape[0]),
                    np.zeros(backgr.shape[0])))


# In[64]:


X_2_train,X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2, test_size=0.99, random_state =5)


# In[65]:


loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_2_test, y_2_test)


# In[67]:


# Final evaluation of the model for DNN
# Testing Outputs
scores = loaded_model.evaluate(X_2_test, y_2_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[68]:


from sklearn.metrics import roc_curve, auc

decisions = loaded_model.predict(X_2_test)


# Compute ROC curve and area under the curve
fpr, tpr, thresholds = roc_curve(y_2_test, decisions)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid()
# plt.savefig("ROC_curve.png")
plt.show()


# In[69]:


import matplotlib
plt.figure()                                     # new window
matplotlib.rcParams.update({'font.size':14})     # set all font sizes
tTest = loaded_model.predict(X_2_test)
# if hasattr(clf, "decision_function"):
#     tTest = clf.decision_function(X_test)        # if available use decision_function
# else:
#     tTest = clf.predict_proba(X_test)[:,1]       # for e.g. MLP need to use predict_proba
tBkg_1 = tTest[y_2_test==0]
tSig_1 = tTest[y_2_test==1]
nBins = 20
tMin = np.floor(np.min(tTest))
tMax = np.ceil(np.max(tTest))
bins = np.linspace(tMin, tMax, nBins+1)
# plt.title('Multilayer perceptron')
plt.xlabel(' $DNN$', labelpad=3)
plt.ylabel('$Probability density$', labelpad=40)
n, bins, patches = plt.hist(tSig_1, bins=bins, density=True, histtype='step', fill=False, color ='dodgerblue' ,edgecolor = 'blue', hatch = 'XX',label='Tprime_800-900')
n, bins, patches = plt.hist(tBkg_1, bins=bins, density=True, histtype='step', fill=False,color = 'red' ,alpha=0.5, edgecolor = 'green', hatch='++', label = 'All Background')
plt.grid(color = 'b', alpha = 0.5, linestyle = 'dashed')
plt.legend(loc='center')
plt.title(' ')
# plt.savefig('/eos/home-s/sraj/M.Sc._Thesis/Plot_M.Sc._thesis/Plot_with_HDF5_files/''output_TPrime1200_all_background.png')
plt.show()


# ### Testing on the each Tprime seperately in the for of method 2

# ### Testing on Tprime 600GeV

# ### Testing on Tprime 700GeV

# ### Testing on Tprime 800GeV

# ### Testing on Tprime 900GeV

# ### Testing on Tprime 1000GeV

# ### Testing on Tprime 1100GeV

# ### Testing on Tprime 1200GeV

# In[ ]:





# In[ ]:




