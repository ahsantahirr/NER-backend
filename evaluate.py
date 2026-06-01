import spacy
from sklearn.metrics import classification_report, confusion_matrix

from data import TRAIN_DATA

nlp = spacy.load("model")

y_true = []
y_pred = []

for text, annotations in TRAIN_DATA:
    doc = nlp(text)

    # Create a map of character positions → label
    true_entities = {}
    for start, end, label in annotations["entities"]:
        for i in range(start, end):
            true_entities[i] = label

    pred_entities = {}
    for ent in doc.ents:
        for i in range(ent.start_char, ent.end_char):
            pred_entities[i] = ent.label_

    # Compare character by character
    for i in range(len(text)):
        true_label = true_entities.get(i, "O")   # O = no entity
        pred_label = pred_entities.get(i, "O")

        y_true.append(true_label)
        y_pred.append(pred_label)

print("Classification Report:")
print(classification_report(y_true, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))