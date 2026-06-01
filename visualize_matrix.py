import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from data import TRAIN_DATA
import spacy

nlp = spacy.load("model")

y_true = []
y_pred = []

labels = ["PERSON", "LOCATION", "DATE", "O"]

for text, annotations in TRAIN_DATA:
    doc = nlp(text)

    true_map = {}
    for start, end, label in annotations["entities"]:
        for i in range(start, end):
            true_map[i] = label

    pred_map = {}
    for ent in doc.ents:
        for i in range(ent.start_char, ent.end_char):
            pred_map[i] = ent.label_

    for i in range(len(text)):
        y_true.append(true_map.get(i, "O"))
        y_pred.append(pred_map.get(i, "O"))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Plot
plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("NER Confusion Matrix (spaCy Model)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()