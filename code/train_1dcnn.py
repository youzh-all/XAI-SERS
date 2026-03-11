# %%
import time
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from keras import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import lime
from lime.lime_tabular import RecurrentTabularExplainer


def plot_confusion_matrix_percent(y_true, y_pred, classes, title='Normalized Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm_norm, row_sums, where=row_sums != 0) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_percent, cmap=plt.cm.Greens)
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
def model_1DCNN(dataset_folderpath, output_folderpath):
    # Load the dataset
    X_train = np.load(os.path.join(dataset_folderpath, 'X_train.npy'))
    X_test = np.load(os.path.join(dataset_folderpath, 'X_test.npy'))
    y_train = np.load(os.path.join(dataset_folderpath, 'y_train.npy'))
    y_test = np.load(os.path.join(dataset_folderpath, 'y_test.npy'))

    # label encoding
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)

    # Data normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape for 1D CNN input
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    # Build the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0009)

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='loss')

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=7, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Add more epochs and consider using an increased batch size if resource permits
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, callbacks=[early_stopping])

    # plot history - loss, accuracy
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['loss'], label='loss')
    plt.legend()
    plt.show()

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy*100}")

    classes = ['Shigella sonnei','Shigella flexneri','Shigella boydii', 'Shigella dysenteriae', 'EIEC', 'EPEC',  'ETEC', 'EAEC', 'STEC']
    
    # recall, precision, f1-score
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred, target_names=classes))

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
    plot_confusion_matrix_percent(y_test, y_pred, classes=['Shigella sonnei', 'Shigella flexneri', 'Shigella boydii', 'Shigella dysenteriae', 'EIEC', 'EPEC', 'ETEC', 'EAEC', 'STEC'], title='Normalized Confusion Matrix')

    classes = ['Shigella sonnei','Shigella flexneri','Shigella boydii', 'Shigella dysenteriae', 'EIEC', 'EPEC',  'ETEC', 'EAEC', 'STEC']
    # Manually set tick labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=60, ha='right')
    plt.yticks(tick_marks, classes)
    plt.show()

    # Save the model
    if not os.path.exists(output_folderpath):
        os.makedirs(output_folderpath)
    model.save(os.path.join(output_folderpath, 'model_1DCNN.h5'))

    # RecurrentTabularExplainer initialization
    explainer = RecurrentTabularExplainer(X_train, training_labels=y_train, feature_names=["feature"],
                                       discretize_continuous=False, feature_selection='auto', class_names=classes)

    # Calculate average feature importance across multiple samples
    num_samples = len(X_test) # Number of samples to explain
    total_feature_importance = np.zeros(X_train.shape[1])  # Initialize total feature importance array

    for sample_idx in range(num_samples):
        # Visualize Lime explanation
        exp = explainer.explain_instance(X_test[sample_idx].reshape(1, -1), model.predict, num_features=X_train.shape[1])
        feature_importance = np.array(exp.as_list())[:, 1].astype(float)  # Extract feature importance values
        total_feature_importance += feature_importance

    # Compute average feature importance
    average_feature_importance = total_feature_importance / num_samples

    # Plot average feature importance
    num_features = X_train.shape[1]
    feature_names = [f"Feature {i+1}" for i in range(num_features)]

    print(exp.as_list())
    plt.figure(figsize=(12, 6))
    plt.plot(feature_names, average_feature_importance, marker='o')
    plt.title('Average Feature Importance of sample')
    plt.xlabel('Feature')
    plt.ylabel('Average Importance')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # Convert exp.as_list() result to a pandas DataFrame
    feature_importance_df = pd.DataFrame(exp.as_list(), columns=['Feature', 'Importance'])

    # Save DataFrame to an Excel file
    excel_file_path = os.path.join(output_folderpath, 'feature_importance.xlsx')
    feature_importance_df.to_excel(excel_file_path, index=False)

    print("Feature importance saved to Excel file:", excel_file_path)

#%%
model_1DCNN('dataset_BCDSpBN', 'outputs_BCDSpBN')

# %% Validation accuracy, loss 추가
def model_1DCNN(dataset_folderpath, output_folderpath):
    # Load the dataset
    X_train = np.load(os.path.join(dataset_folderpath, 'X_train.npy'))
    X_test = np.load(os.path.join(dataset_folderpath, 'X_test.npy'))
    y_train = np.load(os.path.join(dataset_folderpath, 'y_train.npy'))
    y_test = np.load(os.path.join(dataset_folderpath, 'y_test.npy'))

    # label encoding
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)

    # Data normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape for 1D CNN input
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    # Build the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0009)

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='loss')

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=7, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Add more epochs and consider using an increased batch size if resource permits
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, callbacks=[early_stopping], validation_data=(X_test, y_test))

    # plot history - loss, accuracy
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['loss'], label='loss')
    plt.legend()
    plt.show()

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy*100}")
    # plot training history - loss, accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
#%%
model_1DCNN('dataset_BCDSpBN', 'outputs_BCDSpBN')
# %%
