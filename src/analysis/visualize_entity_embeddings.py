import logging
import src.util.Util as Util
import src.util.Training as Training
import src.models.ClaimDetection as CD_Models
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger("Cross-lingual-claim-detection")

task = "verifiable-claim-detection"
config = Util.readAllConfig()
path = config["path"]
claim_detection_config = config[task]
model_name = claim_detection_config["default-model"]
max_length = claim_detection_config["max-length"]  # Refer BertTweet Paper

# Common training parameters
hidden_size = 256
output_size = 2
MODELS_PATH = path + "Models/Claim-Detection/"
RESULTS_PATH = path + "Results/Claim-Detection/Images/"
iterations = 1
we_size = 768
ee_size = 128
run_name = "-XLMR"

no_entityC = 5
no_entityCW = 9
no_entityF = 16
no_entityFW = 31

wiki_entity_presence_threshold = -0.15


def visualizeEntityTypeEmbeddings(embeddings, name, labels, perplexity, no_of_clusters):
    weights = embeddings.weight.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=no_of_clusters, random_state=42)
    clusters = kmeans.fit_predict(weights)
    silhouette_avg = silhouette_score(weights, clusters)
    print(f"Silhouette score for the cluster size {no_of_clusters} - {silhouette_avg}")

    similarity_matrix = cosine_similarity(weights)
    plt.figure(figsize=(20, 20))
    sns.heatmap(similarity_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap='magma_r')
    plt.savefig(RESULTS_PATH + name + "entity_type_similarity.png")
    plt.show()

    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
    reduced_weights = tsne.fit_transform(weights)
    colors = plt.cm.get_cmap('rainbow', no_of_clusters)

    plt.figure(figsize=(12, 8))

    offset = 1
    for e, word in enumerate(labels):
        plt.scatter(reduced_weights[e, 0], reduced_weights[e, 1], color=colors(clusters[e]))
        plt.text(reduced_weights[e, 0] + offset, reduced_weights[e, 1] + offset, word, fontsize=12)

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors(c), markersize=12,
                          label=f"Cluster {c}") for c in range(no_of_clusters)]
    plt.legend(handles=handles, loc='upper left', fontsize=12)
    plt.savefig(RESULTS_PATH + name + "_entity_type_embeddings_C" + str(no_of_clusters) + ".png")
    plt.show()

"""
########################################################################################################################
# Word embedding + Coarse-grained NER
model_prefix = "EClassifier-cNER-" + run_name
logging.info(f"Started training {model_prefix}")

for i in range(iterations):
    model_name = model_prefix + "-" + str(i) + ".pth"
    model = CD_Models.EClassifier(we_size, ee_size, no_entityC, hidden_size, output_size)
    best_model = Training.loadModel(model, MODELS_PATH + model_name)
    entity_embedding = best_model.entity_embedding
    visualizeEntityTypeEmbeddings(entity_embedding, model_name,
                                  ["Other", "Person", "Organization", "Location", "Miscellaneous"], 2, 3)

logging.info(f"Completed training {model_prefix}")

########################################################################################################################
"""
# Clear all the variables
model = None
best_model = None
Util.clearMemory()

########################################################################################################################
ee_size = 256
# Word embedding + NER + Wiki Entity Presence
model_prefix = "EClassifier-NER-cWiki" + str(int(wiki_entity_presence_threshold * 100)) + "-E256" + run_name
logging.info(f"Started training {model_prefix}")

for i in range(iterations):
    model_name = model_prefix + "-" + str(i) + ".pth"
    model = CD_Models.EXClaim(we_size, ee_size, no_entityCW, hidden_size, output_size)
    best_model = Training.loadModel(model, MODELS_PATH + model_name)
    entity_embedding = best_model.entity_embedding
    visualizeEntityTypeEmbeddings(entity_embedding, model_name,
                                  ["Other", "Person", "Organization", "Location", "Miscellaneous", "Imp-Person",
                                   "Imp-Organization", "Imp-Location", "Imp-Miscellaneous "], 3, 3)

########################################################################################################################
# Coarse-grained NER models completed. Clear all the variables

# Clear all the variables
model = None
best_model = None
Util.clearMemory()

########################################################################################################################
ee_size = 128
# Word embedding + Fine-grained NER
model_prefix = "EClassifier-fNER-" + run_name
logging.info(f"Started training {model_prefix}")

for i in range(iterations):
    model_name = model_prefix + "-" + str(i) + ".pth"
    model = CD_Models.EXClaim(we_size, ee_size, no_entityF, hidden_size, output_size)
    best_model = Training.loadModel(model, MODELS_PATH + model_name)
    entity_embedding = best_model.entity_embedding
    visualizeEntityTypeEmbeddings(entity_embedding, model_name,
                                  ["Other", "Person", "Organization", "Location", "Animal", "Biological entity",
                                   "Celestial Body", "Disease", "Event",  "Food", "Instrument",  "Media",
                                   "Mythological entity", "Plant", "Time", "Vehicle"], 4, 7)
"""
########################################################################################################################
ee_size = 256
# Word embedding + NER + Wiki Entity Presence
model_prefix = "EClassifier-NER-fWiki" + str(int(wiki_entity_presence_threshold * 100)) + "-" + run_name
logging.info(f"Started training {model_prefix}")

for i in range(iterations):
    model_name = model_prefix + "-" + str(i) + ".pth"
    model = CD_Models.EClassifier(we_size, ee_size, no_entityFW, hidden_size, output_size)
    best_model = Training.loadModel(model, MODELS_PATH + model_name)
    entity_embedding = best_model.entity_embedding
    visualizeEntityTypeEmbeddings(entity_embedding, model_name,
                                  ["Other", "Person", "Organization", "Location", "Animal", "Biological entity",
                                   "Celestial Body", "Disease", "Event", "Food", "Instrument", "Media",
                                   "Mythological entity", "Plant", "Time", "Vehicle", "Imp-Person", "Imp-Organization",
                                   "Imp-Location", "Imp-Animal", "Imp-Biological entity", "Imp-Celestial Body",
                                   "Imp-Disease", "Imp-Event", "Imp-Food", "Imp-Instrument", "Imp-Media",
                                   "Imp-Mythological entity", "Imp-Plant", "Imp-Time", "Imp-Vehicle"], 5, 7)
"""