import pandas as pd
import src.util.Util as Util
import src.util.Results as Results
from sklearn.metrics import confusion_matrix

def majority_prediction(row):
    return row.value_counts().idxmax()

config = Util.readAllConfig()
path = config["path"]
RESULTS_PATH = path + "Results/Claim-Detection/"

test_data = pd.read_csv(path + "Data/All/combined_test_predictions.csv")
models = Results.getModelNames(test_data)


for model in models:
    model_columns = [model + "-" + str(i) for i in range(10)]
    test_data[model] = test_data[model_columns].apply(majority_prediction, axis=1)
    test_data = test_data.drop(columns=model_columns)

print("Majority prediction taken. Columns now - ", test_data.columns)

sampled_df = test_data.groupby('language').apply(lambda x: x.sample(250)).reset_index(drop=True)

sampled_df.to_csv(RESULTS_PATH + "sample.csv")

results = []

for model in models:
    cm = confusion_matrix(sampled_df['class_label'], sampled_df[model])
    results.append({'Model': model, 'TN': cm[0][0], 'FN': cm[0][1], 'FP': cm[1][0], 'TP': cm[1][1]})

df = pd.DataFrame(results).sort_values(by="Model", ascending=False)
df.to_csv(RESULTS_PATH + "confusion_matrix.csv")

