# tensorinjo = "tensorflow-2.11.0-cp38-cp38-win_amd64.whl"
# stelargrfinjo = "stellargraph-1.2.1-py3-none-any.whl"
#
# import pip
#
# def install_whl(path):
#    pip.main(['install', path])
#
# import math
#
# install_whl(tensorinjo)
# install_whl(stelargrfinjo)
import math
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, f1_score, \
    matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split
from stellargraph import datasets
from stellargraph.layer import DeepGraphCNN
from stellargraph.mapper import PaddedGraphGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
import seaborn as sns
dataset = datasets.PROTEINS()
graphs, graph_labels = dataset.load()

print("______________________________________")
print(graphs[0].info())
print("______________________________________")
print(graphs[1].info())

summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
    columns=["nodes", "edges"],
)
corr_matrix = summary.corr()




plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Heatmap of Correlation between Nodes and Edges in PROTEINS Dataset')
plt.show()
print("______________________________________")
print(summary.describe().round(1))

print("______________________________________")
print(graph_labels.values)

print("______________________________________")
print(graph_labels.value_counts().to_frame())
print("______________________________________")
graph_labels = pd.get_dummies(graph_labels, drop_first=True)
print(graph_labels.value_counts().to_frame())

generator = PaddedGraphGenerator(graphs=graphs)

k = 35
layer_sizes = [32, 32, 32, 1]

dgcnn_model = DeepGraphCNN(
    layer_sizes=layer_sizes,
    activations=["tanh", "tanh", "tanh", "tanh"],
    k=k,
    bias=False,
    generator=generator,
)
x_inp, x_out = dgcnn_model.in_out_tensors()

x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
x_out = MaxPool1D(pool_size=2)(x_out)

x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

x_out = Flatten()(x_out)

x_out = Dense(units=128, activation="relu")(x_out)
x_out = Dropout(rate=0.5)(x_out)

predictions = Dense(units=1, activation="sigmoid")(x_out)

model = Model(inputs=x_inp, outputs=predictions)

model.compile(
    optimizer=Adam(lr=0.0001), loss=binary_crossentropy, metrics=["acc"],
)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

cv = StratifiedKFold(n_splits=10, shuffle=True)

epochs = 100
gm_values = []
precision_values = []
recall_values = []
f1_values = []
roc_auc_values = []
fpr_values = []
tpr_values = []
mcc_values = []
histories = []


def rest_of_metrics(y_true, y_pred):
    y_pred = K.cast(K.round(y_pred), K.floatx())
    y_true = K.cast(y_true, K.floatx())

    y_pred_np = K.eval(y_pred)  # Convert y_pred tensor to NumPy array
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_np).ravel()

    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    gm = math.sqrt(tpr * tnr)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("GM: ", gm)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    gm_values.append(K.get_value(gm))
    precision_values.append(K.get_value(precision))
    recall_values.append(K.get_value(recall))
    f1_values.append(K.get_value(f1))
    fpr_values.append(K.get_value(fpr))
    tpr_values.append(K.get_value(tpr))


def roc_auc_metric(y_true, y_pred):
    y_pred = K.cast(K.round(y_pred), K.floatx())
    y_true = K.cast(y_true, K.floatx())

    roc_auc = roc_auc_score(y_true, y_pred)
    print("ROC AUC: ", roc_auc)
    roc_auc_values.append(K.get_value(roc_auc))


def mcc_metric(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_pred = y_pred.round()
    mcc = matthews_corrcoef(y_true, y_pred)
    print("MCC: ", mcc)
    mcc_values.append(mcc)


for train_index, test_index in cv.split(graphs, graph_labels):
    graphs = np.array(graphs)
    X_train, X_test = graphs[train_index.astype(int)], graphs[test_index.astype(int)]
    y_train, y_test = graph_labels.iloc[train_index.astype(int)], graph_labels.iloc[test_index.astype(int)]
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    gen = PaddedGraphGenerator(graphs=graphs)

    train_gen = gen.flow(
        X_train,
        targets=y_train,
        batch_size=32,
        symmetric_normalization=False,
    )

    val_gen = gen.flow(
        X_val,
        targets=y_val,
        batch_size=50,
        symmetric_normalization=False,
    )

    test_gen = gen.flow(
        X_test,
        targets=y_test,
        batch_size=50,
        symmetric_normalization=False,
    )

    history = model.fit(
        train_gen, epochs=epochs, verbose=1, validation_data=val_gen, shuffle=True, callbacks=[callback]
    )

    histories.append(history)

    y_pred = model.predict(test_gen)
    y_pred = np.reshape(y_pred, (-1,))
    roc_auc_metric(y_test, y_pred)
    y_pred = [0 if prob < 0.5 else 1 for prob in y_pred]

    y_test = y_test.to_numpy()
    y_test = np.reshape(y_test, (-1,))

    rest_of_metrics(y_test, y_pred)
    mcc_metric(y_test, y_pred)

data = {
    "Metric": ["MCC", "GM", "Precision", "Recall", "F1", "ROC AUC", "FPR", "TPR"],
    "Average": [np.mean(mcc_values), np.mean(gm_values), np.mean(precision_values), np.mean(recall_values),
                np.mean(f1_values), np.mean(roc_auc_values), np.mean(fpr_values), np.mean(tpr_values)],
    "Maximum": [np.max(mcc_values), np.max(gm_values), np.max(precision_values), np.max(recall_values),
                np.max(f1_values), np.max(roc_auc_values), np.max(fpr_values), np.max(tpr_values)],
    "Minimum": [np.min(mcc_values), np.min(gm_values), np.min(precision_values), np.min(recall_values),
                np.min(f1_values), np.min(roc_auc_values), np.min(fpr_values), np.min(tpr_values)]
}

df = pd.DataFrame(data)

#df.to_excel("metrics.xlsx", index=False)
# plt.figure(figsize=(10, 8))
# sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Performance Metrics Heatmap')
# plt.show()

display(df)
df = df.set_index('Metric')

# Transpose the dataframe to make the metrics as columns for the heatmap
df_t = df.T

df_metrics = pd.DataFrame({
    'MCC': mcc_values,
    'GM': gm_values,
    'Precision': precision_values,
    'Recall': recall_values,
    'F1': f1_values,
    'ROC AUC': roc_auc_values,
    'FPR': fpr_values,
    'TPR': tpr_values
})

# Creating the boxplot for each performance metric
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_metrics)
plt.title('Boxplot of Performance Metrics')
plt.ylabel('Metric Value')
plt.xlabel('Metric')
plt.show()

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(history.history["acc"])
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.subplot(1, 3, 2)
plt.plot(history.history["loss"])
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 3, 3)
plt.plot(history.history["val_acc"])
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")


cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

# plt.figure(figsize=(8, 6))  # Optional: Set the size of the plot
# sns.heatmap(df, annot=True, cmap='coolwarm', fmt='.1f', linewidths=0.5)
# plt.title('Heatmap of DataFrame')
# plt.show()

# print(predictions)


# Transpose the DataFrame to make the metrics as columns for the heatmap


# Create a heatmap
# plt.figure(figsize=(10, 6))
# heatmap = sns.heatmap(df_t, annot=True, cmap='viridis')
# plt.title('Heatmap of Metric Values')
# plt.show()


sns.pairplot(summary)
plt.show()

# sns.pairplot(pd.DataFrame(dataset))
# plt.show()
# Create a heatmap
if 'Metric' in df.columns:
    df = df.set_index('Metric')

# Transpose the dataframe for the heatmap
df_t = df.T

# Create the heatmap
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(df_t, annot=True, cmap='viridis', fmt=".2f")
plt.title('Heatmap of Metric Values')
plt.show()