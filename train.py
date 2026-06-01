import spacy
from spacy.training.example import Example
from data import TRAIN_DATA

nlp = spacy.load("en_core_web_sm")

# Add NER pipe if not exists
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Add labels
for _, annotations in TRAIN_DATA:
    for ent in annotations["entities"]:
        ner.add_label(ent[2])

optimizer = nlp.resume_training()

for epoch in range(10):  # small training
    losses = {}

    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.2, losses=losses)

    print("Loss:", losses)

nlp.to_disk("model")
print("Model saved!")