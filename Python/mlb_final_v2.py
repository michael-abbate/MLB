'''
mlb_final.py includes a Decision Tree, XGBoost Classifier, Random Forest, Voting Classifier, and Neural Network
    to predict at-bat outcomes.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow import keras
import time
from datetime import datetime


# Performance Report
def evaluate(y_test, y_pred):
    print("Confusion Matrix: ")
    # print(y_test.value_counts())
    print(confusion_matrix(y_test, y_pred))

    print ("Accuracy : ")
    print(accuracy_score(y_test,y_pred)*100)

    print("Report : ")
    report = classification_report(y_test, y_pred)
    print(report)

def main():
    # Get start time of the process
    start_time = datetime.now()
    current_time = start_time.strftime("%H:%M:%S")
    print("Time start:", current_time)
    print()
    start = time.time()

    # Preprocess the data
    baseball_orig = pd.read_csv('mlb_cleaned_updated_PLOC.csv')
    b_nan = sum(baseball_orig.isnull().sum(axis = 0))
    baseball_orig.dropna(inplace=True)
    print("The total number of rows is:", len(baseball_orig))
    print("The number of rows due to NaN dropped:", b_nan, '\n')
    print('Initial Value Counts:')
    print(baseball_orig.event.value_counts(), '\n')

    # Separate majority and minority classes
    df_majority = baseball_orig[baseball_orig.event == 1]
    df_minority = baseball_orig[baseball_orig.event == 0]

    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,     # sample with replacement
                                     n_samples= baseball_orig['event'].sum(),    # to match majority class
                                     random_state=123) # reproducible results
    # Combine majority class with upsampled minority class
    baseball = pd.concat([df_majority, df_minority_upsampled])

    # Display new class counts
    print('New Value Counts After Resampling:')
    print(baseball.event.value_counts(), '\n')

    # Normalize the columns necessary using fct: (X-X.min)/(X.max - X.min)
    X_orig = baseball.iloc[:, 7:-1]
    for col in X_orig:
        if X_orig[str(col)].max() !=1:
            col = str(col)
            X_orig[col] = (X_orig[col] - X_orig[col].min()) / (X_orig[col].max() - X_orig[col].min())
    X = X_orig

    # Get outcome variable
    y = baseball.iloc[:, -1:]

    # Split data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.3, random_state = 100)
    print("The shape of X_train: ", X_train.shape)
    print("The shape of y_train: ", y_train.shape)
    print("The shape of X_test: ", X_test.shape)
    print("The shape of y_test: ", y_test.shape, '\n')


    # Set the classifiers with correct parameters
    tree_classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 100)

    xgb_classifier = xgb.XGBClassifier(max_depth = 15)

    rf_classifier = RandomForestClassifier(criterion = 'gini', random_state = 100,    #max_features='sqrt',
                                          bootstrap=True, class_weight='balanced',
                                           n_estimators=245)


    # Fit, evaluate, get important features, and plot all ROC Curves together
    classifiers = [tree_classifier, xgb_classifier, rf_classifier]
    result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

    # Train the models and record the results
    for cls in classifiers:
        # Get classifier name
        name = str(cls.__class__.__name__)
        print(name, '--------------------------------------------------')

        # Fit and predict
        model = cls.fit(X_train, y_train)
        yproba = model.predict_proba(X_test)[::,1]
        y_pred = cls.predict(X_test)
        evaluate(y_test, y_pred)
        print()

        # Importances
        imp = pd.DataFrame(cls.feature_importances_, index=X_train.columns,
                                columns = ['importance']).sort_values('importance',ascending=False)
        print(imp)
        print()

        # Get ROC Values
        fpr, tpr, _ = roc_curve(y_test,  yproba)
        auc = roc_auc_score(y_test, yproba)
        result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                            'fpr':fpr,
                                            'tpr':tpr,
                                            'auc':auc}, ignore_index=True)


    # Actual plotting of the ROC Curves graph:

    # Set name of the classifiers as index labels
    result_table.set_index('classifiers', inplace=True)

    fig = plt.figure(figsize=(8, 6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'],
                 result_table.loc[i]['tpr'],
                 label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')

    plt.show()



    # Get metrics for Random Forest with threshold = 0.82
    print('Random Forest with 0.82 Threshold --------------------------------------------------')
    y_pred_thresh = (rf_classifier.predict_proba(X_test)[:,1] >= 0.82).astype(bool)
    evaluate(y_test, y_pred_thresh)
    print()


    # Fit and Evaluate the Voting Classifier
    vt_classifier = VotingClassifier(estimators = [('tree', tree_classifier), ('rf', rf_classifier),
                                                   ("xgb", xgb_classifier)])
    print(VotingClassifier.__class__.__name__, '--------------------------------------------------')
    vt_classifier.fit(X_train, y_train)
    y_pred = vt_classifier.predict(X_test)
    evaluate(y_test, y_pred)
    print()


    ####################################################################################
    # Neural Network
    print('Neural Network --------------------------------------------------')

    model = keras.models.Sequential()
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(250, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(175, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(75, activation='relu'))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # Print the summary of the network structure
    #print(model.summary())

    my_adam = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=my_adam, metrics=['accuracy'])



    # Use the last 10% of X_train as validation data
    checkpointer = keras.callbacks.ModelCheckpoint("best_dnn_so_far.h5")
    earlystopper = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=90, validation_split=0.1, batch_size= 64,
                        callbacks=[checkpointer, earlystopper])

    print("--------------Train:--------------")
    model.evaluate(X_train, y_train)
    print("--------------Test:---------------")
    model.evaluate(X_test, y_test)

    # plot loss over time
    plt.plot(history.history['loss'])
    plt.show()


    ####################################################################################

    print()
    print('Time to complete:', (time.time()-start)/60, 'minutes')



if __name__=="__main__":
    main()
