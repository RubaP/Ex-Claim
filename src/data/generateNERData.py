import pandas as pd
import src.util.Util as Util
import src.util.NER as NER
import src.util.Preprocess as Preprocess
import numpy as np

config = Util.readAllConfig()
path = config["path"]

test_data = pd.read_csv(path + "Data/All/combined_test.csv")
test_data['preprocessed_text'] = test_data['tweet_text'].apply(lambda x: Preprocess.processTweetText(x))
X = list(test_data['preprocessed_text'])
number_of_sentences = len(X)

cNER = ["cPER", "cORG", "cLOC", "cMISC"]
cNER_model = "Babelscape/wikineural-multilingual-ner"
ner_pipeline, NER_TAGS = NER.readPipeline(path, cNER_model)
NER_labels = NER.getNER(ner_pipeline, X)
ner_vectors = []

for sentence_id in range(number_of_sentences):
    sentence_tags = NER_labels[sentence_id]
    ner_vector = [0]*len(cNER)

    for ner_label in sentence_tags:
        entity = ner_label['entity']
        tag_id = NER_TAGS[entity]
        if tag_id > 0 and (tag_id%2) == 1:
            entity_id = int((tag_id-1)/2)
            ner_vector[entity_id] +=1

    ner_vectors.append(ner_vector)

ner_vectors = np.array(ner_vectors)

for i in range(len(cNER)):
    test_data[cNER[i]] = ner_vectors[:,i]

cNER_Exits = np.any(ner_vectors > 0, axis=1)
test_data["cNER_Exists"] = cNER_Exits.tolist()


fNER_model = "multinerd-mbert"
fNER = ["fPER", "fORG", "fLOC", "fANIM", "fBIO", "fCEL", "fDIS", "fEVE", "fFOOD", "fINST", "fMEDIA", "fMYTH",
        "fPLANT", "fTIME", "fVEHI"]
ner_pipeline, NER_TAGS = NER.readPipeline(path, fNER_model)
NER_labels = NER.getNER(ner_pipeline, X)
ner_vectors = []

for sentence_id in range(number_of_sentences):
    sentence_tags = NER_labels[sentence_id]
    ner_vector = [0]*len(fNER)

    for ner_label in sentence_tags:
        entity = ner_label['entity']
        tag_id = NER_TAGS[entity]
        if tag_id > 0:
            entity_id = int((tag_id - 1) / 2)
            ner_vector[entity_id] += 1

    ner_vectors.append(ner_vector)

ner_vectors = np.array(ner_vectors)

for i in range(len(fNER)):
    test_data[fNER[i]] = ner_vectors[:,i]

fNER_Exits = np.any(ner_vectors > 0, axis=1)
test_data["fNER_Exists"] = fNER_Exits.tolist()

test_data.to_csv(path + "Data/All/combined_test_ner.csv", index=False)
