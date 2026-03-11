#%%
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


def plot_confusion_matrix_percent(y_true, y_pred, classes, title='Normalized Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm_norm, row_sums, where=row_sums != 0) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_percent, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=60, ha='right')
    ax.set_yticklabels(classes)

    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            ax.text(j, i, f'{cm_percent[i, j]:.2f}%', ha='center', va='center', fontsize=7)

    plt.tight_layout()
    plt.show()

#%%
def model_RF(dataset_folderpath, output_folderpath):
    # Load the dataset
    X_train = np.load(os.path.join(dataset_folderpath, 'X_train.npy'))
    X_test = np.load(os.path.join(dataset_folderpath, 'X_test.npy'))
    y_train = np.load(os.path.join(dataset_folderpath, 'y_train.npy'))
    y_test = np.load(os.path.join(dataset_folderpath, 'y_test.npy'))

    # Data normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_model.fit(X_train, y_train)

    # Model evaluation
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test accuracy: {accuracy}')

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f'Test Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    # Confusion Matrix Visualization (percent format, 2 decimals)
    plot_confusion_matrix_percent(y_test, y_pred, classes=['EIEC_BCDSpBN', 'Shigella sonnei_BCDSpBN', 'Shigella dysenteriae_BCDSpBN', 'EPEC_BCDSpBN', 'Shigella boydii_BCDSpBN', 'Shigella flexneri_BCDSpBN', 'ETEC_BCDSpBN', 'EAEC_BCDSpBN', 'STEC_BCDSpBN'], title='Normalized Confusion Matrix')


    classes = ['EIEC_BCDSpBN', 'Shigella sonnei_BCDSpBN', 'Shigella dysenteriae_BCDSpBN', 'EPEC_BCDSpBN', 'Shigella boydii_BCDSpBN', 'Shigella flexneri_BCDSpBN', 'ETEC_BCDSpBN', 'EAEC_BCDSpBN', 'STEC_BCDSpBN']
   # Manually set tick labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=60, ha='right')
    plt.yticks(tick_marks, classes)
    plt.show()

#%%
model_RF('dataset_BCDSpBN', 'outputs_BCDSpBN')
# %%
