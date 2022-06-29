## Analysis Stratigies
#### Model Preprations:-
1. For the training of the Deep Neural Network(DNN), the ntuples have been divided for training and validation samples.
2. for the training of the DNN, the Keras Sequential model have been used.<br />
**DNN Model**
```
clf = Sequential()
# clf.add(LSTM(1, return_sequences=True ))
clf.add(BatchNormalization(input_shape = (42,)))
initializer =tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
# clf.add(Dropout(0.3))
clf.add(Dense(512, activation = 'relu',kernel_regularizer=regularizers.l2(0.001), kernel_initializer = 'random_uniform' ,name = 'dense_11'))
clf.add(BatchNormalization())
clf.add(Dropout(0.3))
clf.add(Dense(512, activation = 'relu',kernel_regularizer=regularizers.l2(0.001), kernel_initializer = 'random_uniform' ,name = 'dense_1'))
clf.add(BatchNormalization())
clf.add(Dropout(0.3))
# clf.add(Dense(512, activation = 'relu',kernel_regularizer=regularizers.l2(0.001), kernel_initializer = 'random_uniform', name = 'dense_2'))
# clf.add(BatchNormalization())
# clf.add(Dropout(0.35))
clf.add(Dense(256, activation = 'relu',kernel_regularizer=regularizers.l2(0.001), kernel_initializer = 'random_uniform' ,name = 'dense_22'))
clf.add(BatchNormalization())
clf.add(Dropout(0.35))
# clf.add(Dense(256, activation = 'relu',kernel_regularizer=regularizers.l2(0.001), kernel_initializer = 'random_uniform', name = 'dense_3'))
# clf.add(BatchNormalization())
# clf.add(Dropout(0.40))
# clf.add(Dense(128, activation = 'relu',kernel_regularizer=regularizers.l2(0.001), kernel_initializer = 'random_uniform' ,name = 'dense_4'))
# clf.add(BatchNormalization())
# clf.add(Dropout(0.40))
clf.add(Dense(128, activation = 'relu',kernel_regularizer=regularizers.l2(0.001), kernel_initializer = 'random_uniform', name = 'dense_5'))
clf.add(BatchNormalization(momentum=0.99,epsilon=0.001,beta_initializer=RandomNormal(mean=0.0, stddev=0.05),gamma_initializer=Constant(value=0.9)))
clf.add(Dense(64, activation = 'relu',kernel_regularizer=regularizers.l2(0.001), kernel_initializer = 'random_uniform', name = 'dense_6'))
clf.add(BatchNormalization(momentum=0.99,epsilon=0.001,beta_initializer=RandomNormal(mean=0.0, stddev=0.05),gamma_initializer=Constant(value=0.9)))
clf.add(Dense(32, activation = 'relu', name = 'dense_7'))
clf.add(Dropout(0.45))
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
# plot_model(clf, to_file='/eos/user/s/sraj/M.Sc._Thesis/Plot_M.Sc._thesis/DNN/600-700/''clf_plot_model___.pdf', show_shapes=True, show_layer_names=True)
```
Then run the model and save the best model.
 - The model have been tested over the NRB background and the signls.
 - Applied blinded region for the diphoton in mass range of 115 GeV to 135 GeV
 - The weight is applied and the model have been plotted again as the stacked plots
 - The data and monte carlo have been scaled with the linear fitting values and data have been sclaed accordingly. 
 - The data have been filled with the scaled inputs.
 - On each mass points, the upper limit with 95% CL is calculated.

