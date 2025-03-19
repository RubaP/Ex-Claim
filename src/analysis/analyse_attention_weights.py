import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import src.util.Util as Util
import src.util.Embedding as Embedding
import src.util.Training as Training
import src.models.ClaimDetection as CD_Models
import src.util.NER as NER
import src.util.EntityLinking as EntityLinking
from genre.trie import Trie, MarisaTrie
import pandas as pd
import torch

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib.colorbar').setLevel(logging.ERROR)
logging.getLogger('matplotlib.pyplot').setLevel(logging.ERROR)

task = "verifiable-claim-detection"
config = Util.readAllConfig()
path = config["path"]
claim_detection_config = config[task]
model_name = claim_detection_config["default-model"]
max_length = claim_detection_config["max-length"]  # Refer BertTweet Paper

texts = ["The Libertarian Party 'is the third-largest political party in the U.S.'",
         "Every year, more than $40 million are leaving Missouri for the Lone Star State.",
         "Old video of cop beating woman in Gwalior shared as Delhi police brutality",
         "Mark Cuban says Twitter 'reaches only 15% of the population'",
         "Donald Trump fundraising email takes CNN anchor's comments out of context",
         "TikTok was still operational in Sri Lanka in early July 2020, and the government says it has no plans to ban the platform",
         "Trump withdraws US from the Iran nuclear deal. Here's what you need to know",
         "World health organisation Giving every government 1.5 lacs after every COVID patient",
         "Today, FDA approved the first COVID-19 vaccine for the prevention of #COVID19 disease in individuals 16 years of age and older.",
         "In Verona, Italy tonight, protestors chant that they will never give up and will keep fighting to end the COVID vaccine mandate."]

tokenized_text = [['The', 'Liber', 'tarian', 'Party', "'", 'is', 'the', 'third', '-', 'lar', 'g', 'est', 'political', 'party', 'in', 'the', 'U', '.', 'S', '.', "'"],
                  ['Every', 'year', ',', 'more', 'than', '$', '40', 'million', 'are', 'leaving', 'Missouri', 'for', 'the', 'Lo', 'ne', 'Star', 'State', '.'],
                  ['Old', 'video', 'of', 'cop', 'be', 'ating', 'woman', 'in', 'G', 'wali', 'or', 'shared', 'as', 'Delhi', 'police', 'brutal', 'ity'],
                  ['Mark', 'Cuba', 'n', 'says', 'Twitter', "'", 'reach', 'es', 'only', '15%', 'of', 'the', 'population', "'"],
                  ['Donald', 'Trump', 'fund', 'rais', 'ing', 'email', 'takes', 'CNN', 'an', 'chor', "'", 's', 'comments', 'out', 'of', 'context'],
                  ['Tik', 'T', 'ok', 'was', 'still', 'operation', 'al', 'in', 'Sri', 'Lanka', 'in', 'early', 'July', '2020', ',', 'and', 'the', 'government', 'says', 'it', 'has', 'no', 'plans', 'to', 'ban', 'the', 'platform'],
                  ['Trump', 'withdraw', 's', 'US', 'from', 'the', 'Iran', 'nuclear', 'deal', '.', 'Here', "'", 's', 'what', 'you', 'need', 'to', 'know'],
                  ['World', 'health', 'organisation', 'Gi', 'ving', 'every', 'government', '1.5', 'la', 'cs', 'after', 'every', 'CO', 'VID', 'patient'],
                  ['Today', ',', 'F', 'DA', 'approved', 'the', 'first', 'CO', 'VID', '-19', 'vaccin', 'e', 'for', 'the', 'pre', 'vention', 'of', '#', 'CO', 'VID', '19', 'disease', 'in', 'individuals', '16', 'years', 'of', 'age', 'and', 'older', '.'],
                  ['In', 'Verona', ',', 'Italy', 'tonight', ',', 'protest', 'ors', 'chant', 'that', 'they', 'will', 'never', 'give', 'up', 'and', 'will', 'keep', 'fighting', 'to', 'end', 'the', 'CO', 'VID', 'vaccin', 'e', 'mandat', 'e', '.']]


data = pd.DataFrame(texts, columns=["preprocessed_text"])
data['class_label'] = 1
dataset = {"english": {"test": data}}

# Get embedding representation
tokenized_input, class_labels, word_embeddings = Embedding.getEmbeddedDataset(path, model_name, dataset, max_length)

# Common training parameters
hidden_size = 256
output_size = 2
MODELS_PATH = path + "Models/Claim-Detection/"
RESULTS_PATH = path + "Results/Claim-Detection/Images/"
iterations = 1
we_size = Training.getInputSize(word_embeddings)
ee_size = 128
run_name = "-XLMR"

word_importances = []

def getAttentionWeightsWithEntity(model, model_name, input_vectors, entity_indexes):
    logging.info(f"Model - {model_name}")
    device = Util.getDevice()
    model.to(device)
    model.eval()

    X = input_vectors["english"]["test"]
    E = entity_indexes["english"]["test"]

    we = X.to(device)
    e = E.to(device)
    with torch.no_grad():
        ee = model.entity_embedding(e)
        w_projected = model.project1(we)
        e_projected = model.project2(ee)
        x = w_projected + e_projected
        context_vector, weights = model.attention(x, x, x)

    logging.info(f"Attention weight size: {weights.shape}")
    return weights.cpu().numpy()

def getAttentionWeights(model, model_name, input_vectors):
    logging.info(f"Model - {model_name}")
    device = Util.getDevice()
    model.to(device)
    model.eval()

    X = input_vectors["english"]["test"]

    we = X.to(device)
    with torch.no_grad():
        x = model.projection(we)
        context_vector, weights = model.attention(x, x, x)

    logging.info(f"Attention weight size: {weights.shape}")
    return weights.cpu().numpy()


def plot_graph(s_weights, s_tokens, custom_cmap, sentence_no, model):
    plt.figure(figsize=(10, 10))
    sns.heatmap(s_weights, xticklabels=s_tokens, yticklabels=s_tokens, annot=False, cmap=custom_cmap)
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.savefig(RESULTS_PATH + model + "_attention_weights_" + str(sentence_no) + ".png")
    plt.clf()

def saveWordImportance(s_weights, s_tokens, j, model):
    global word_importances

    average_weights = np.sum(s_weights, axis=0)
    word_weights = average_weights[1: len(s_tokens) + 1]

    word_importances.append({"index": j, "Model": model, "weights": word_weights})

def visualize_attention_weights(weights, tokens, model):
    size = len(tokens)
    logging.info(f"Number of sentences: {size}")
    custom_cmap = sns.light_palette("navy", reverse=False, as_cmap=True)
    for j in range(size):
        s_tokens = tokens[j]
        s_weights = weights[j][1: len(s_tokens)+1, 1:len(s_tokens)+1]
        logging.info(f"Sliced attention weight shape: {s_weights.shape}")
        logging.info(f"Processing sentence {j} of size {len(s_tokens)}")
        plot_graph(s_weights, s_tokens, custom_cmap, j, model)
        saveWordImportance(weights[j], s_tokens, j, model)

def saveWordImportanceAsExcel(word_importance_list):
    importance_df = pd.DataFrame(word_importance_list)

    excel_writer = pd.ExcelWriter(path + 'Results/Claim-Detection/sample_sentence_importance.xlsx', engine='xlsxwriter')
    for dType, group in importance_df.groupby('index'):
        df = group.drop('index', axis=1).sort_values(by="Model", ascending=False)
        words = tokenized_text[dType]
        weights = list(zip(*df["weights"].tolist()))

        for k, col_data in enumerate(weights):
            df[words[k]] = col_data

        df.to_excel(excel_writer, sheet_name="Sentence-"+str(dType), index=False)
    excel_writer.close()

########################################################################################################################
# Word embedding
model_prefix = "PClassifier" + run_name
logging.info(f"Started training {model_prefix}")
for i in range(iterations):
    model_name = model_prefix + "-" + str(i) + ".pth"
    model = CD_Models.XClaim(we_size, hidden_size, output_size)
    best_model = Training.loadModel(model, MODELS_PATH + model_name)
    attention_weights = getAttentionWeights(best_model, model_name, word_embeddings)
    visualize_attention_weights(attention_weights, tokenized_text, model_prefix)
logging.info(f"Completed evaluating {model_name}")

########################################################################################################################
ner_model = "Babelscape/wikineural-multilingual-ner"
ner_vectors_cg, ner_tagged_sentences_cg = NER.getNERVectors(path, ner_model, dataset, tokenized_input)
entity_indexes, no_entityC = NER.getIndexVector(ner_vectors_cg)

logging.info(f"Entity Index: {entity_indexes['english']['test'][:,:32]}")
########################################################################################################################
# Word embedding + Coarse-grained NER
model_prefix = "EClassifier-cNER-" + run_name
logging.info(f"Started training {model_prefix}")

for i in range(iterations):
    model_name = model_prefix + "-" + str(i) + ".pth"
    model = CD_Models.EXClaim(we_size, ee_size, no_entityC, hidden_size, output_size)
    best_model = Training.loadModel(model, MODELS_PATH + model_name)
    attention_weights = getAttentionWeightsWithEntity(best_model, model_name, word_embeddings, entity_indexes)
    visualize_attention_weights(attention_weights, tokenized_text, model_prefix)
logging.info(f"Completed evaluating {model_name}")

########################################################################################################################

# Clear all the variables
model = None
best_model = None
Util.clearMemory()

########################################################################################################################

# Wiki entity extraction using coarse-grained NER
wiki_entity_scores_cg = EntityLinking.getWikiEntityPresenceScore(path, ner_tagged_sentences_cg, ner_vectors_cg)

wiki_entity_presence_threshold = -0.15

merged = Embedding.concatenateEmbeddings(word_embeddings, ner_vectors_cg)
entity_indexes, no_entityCW = EntityLinking.getELIndexVector(entity_indexes, wiki_entity_scores_cg,
                                                             wiki_entity_presence_threshold, no_entityC)
logging.info(f"Entity Index: {entity_indexes['english']['test'][:,:32]}")
########################################################################################################################
ee_size = 256
# Word embedding + NER + Wiki Entity Presence
model_prefix = "EClassifier-NER-cWiki" + str(int(wiki_entity_presence_threshold * 100)) + "-E256" + run_name
logging.info(f"Started training {model_prefix}")

for i in range(iterations):
    model_name = model_prefix + "-" + str(i) + ".pth"
    model = CD_Models.EXClaim(we_size, ee_size, no_entityCW, hidden_size, output_size)
    best_model = Training.loadModel(model, MODELS_PATH + model_name)
    attention_weights = getAttentionWeightsWithEntity(best_model, model_name, word_embeddings, entity_indexes)
    visualize_attention_weights(attention_weights, tokenized_text, model_prefix)
logging.info(f"Completed evaluating {model_name}")

########################################################################################################################
# Coarse-grained NER models completed. Clear all the variables

# Clear all the variables
model = None
best_model = None
ner_vectors_cg = None
ner_tagged_sentences_cg = None
wiki_entities_cg = None
wiki_entity_scores_cg = None
Util.clearMemory()

########################################################################################################################

ner_model = "multinerd-mbert"
ner_vectors_fg, ner_tagged_sentences_fg = NER.getNERVectors(path, ner_model, dataset, tokenized_input)
entity_indexes, no_entityF = NER.getIndexVector(ner_vectors_fg)
logging.info(f"Entity Index: {entity_indexes['english']['test'][:,:32]}")
########################################################################################################################
ee_size = 128
# Word embedding + Fine-grained NER
model_prefix = "EClassifier-fNER-" + run_name
logging.info(f"Started training {model_prefix}")

for i in range(iterations):
    model_name = model_prefix + "-" + str(i) + ".pth"
    model = CD_Models.EXClaim(we_size, ee_size, no_entityF, hidden_size, output_size)
    best_model = Training.loadModel(model, MODELS_PATH + model_name)
    attention_weights = getAttentionWeightsWithEntity(best_model, model_name, word_embeddings, entity_indexes)
    visualize_attention_weights(attention_weights, tokenized_text, model_prefix)
logging.info(f"Completed evaluating {model_name}")

########################################################################################################################

# Wiki entity extraction using fine-grained NER
wiki_entity_scores_fg = EntityLinking.getWikiEntityPresenceScore(path, ner_tagged_sentences_fg, ner_vectors_fg)

wiki_entity_presence_threshold = -0.15
entity_indexes, no_entityFW = EntityLinking.getELIndexVector(entity_indexes, wiki_entity_scores_fg,
                                                             wiki_entity_presence_threshold, no_entityF)
logging.info(f"Entity Index: {entity_indexes['english']['test'][:,:32]}")
########################################################################################################################
ee_size = 256
# Word embedding + NER + Wiki Entity Presence
model_prefix = "EClassifier-NER-fWiki" + str(int(wiki_entity_presence_threshold * 100)) + "-" + run_name
logging.info(f"Started training {model_prefix}")

for i in range(iterations):
    model_name = model_prefix + "-" + str(i) + ".pth"
    model = CD_Models.EXClaim(we_size, ee_size, no_entityFW, hidden_size, output_size)
    best_model = Training.loadModel(model, MODELS_PATH + model_name)
    attention_weights = getAttentionWeightsWithEntity(best_model, model_name, word_embeddings, entity_indexes)
    visualize_attention_weights(attention_weights, tokenized_text, model_prefix)
logging.info(f"Completed evaluating {model_name}")

saveWordImportanceAsExcel(word_importances)