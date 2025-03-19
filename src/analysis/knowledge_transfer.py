import pandas as pd
import src.util.Util as Util
import src.util.Results as Results

def majority_prediction(row):
    return row.value_counts().idxmax()

config = Util.readAllConfig()
path = config["path"]

test_data = pd.read_csv(path + "Data/All/combined_test_predictions.csv")
models = Results.getModelNames(test_data)

test_data_with_similar_claims = test_data[(test_data["source"] == "synthetic") | (test_data["source"] == "checkthat")]

for model in models:
    model_columns = [model + "-" + str(i) for i in range(10)]
    test_data_with_similar_claims[model] = test_data_with_similar_claims[model_columns].apply(majority_prediction, axis=1)
    test_data_with_similar_claims = test_data_with_similar_claims.drop(columns=model_columns)

print("Majority prediction taken. Columns now - ", test_data_with_similar_claims.columns)

synthetic_claims = test_data_with_similar_claims[test_data_with_similar_claims["source"] == "synthetic"]
checkthat_claims = test_data_with_similar_claims[test_data_with_similar_claims["source"] == "checkthat"]

print("Synthetic Data size: ", synthetic_claims.shape[0])
print("Checkthat Data size: ", checkthat_claims.shape[0])

merged_claims = pd.merge(synthetic_claims, checkthat_claims, on='tweet_id', suffixes=('_syn', '_ct'))


size = merged_claims.shape[0]
print("Merged Data size: ", size)

grouped_by_languages = merged_claims.groupby('language_syn')

results = []

for language, group_df in grouped_by_languages:
    for model in models:
        matching_predictions = (group_df[model+"_syn"] == group_df[model + "_ct"]).sum()
        percentage_matching = (matching_predictions/len(group_df))*100
        results.append({'language': language, 'Type': "All", 'Model': model, 'Percentage': percentage_matching})

        # Matching correct predictions
        group_df_correct = group_df[group_df[model + "_ct"] == group_df['class_label_ct']]
        matching_predictionsC = (group_df_correct[model + "_syn"] == group_df_correct[model + "_ct"]).sum()
        percentage_matchingC = (matching_predictionsC / len(group_df_correct)) * 100
        results.append({'language': language, 'Type': "Correct", 'Model': model, 'Percentage': percentage_matchingC})

        # Matching wrong predictions
        group_df_wrong = group_df[group_df[model + "_ct"] != group_df['class_label_ct']]
        matching_predictionsW = (group_df_wrong[model + "_syn"] == group_df_wrong[model + "_ct"]).sum()
        percentage_matchingW = (matching_predictionsW / len(group_df_wrong)) * 100
        results.append({'language': language, 'Type': "Wrong", 'Model': model, 'Percentage': percentage_matchingW})


percentages_df = pd.DataFrame(results)

#percentages_df.to_csv(path + "Results/Claim-Detection/similar_claims_kt_language_level.csv", index=False)

average_percentages_df = percentages_df.groupby(['Model', 'Type'])['Percentage'].mean().reset_index()

excel_writer = pd.ExcelWriter(path + 'Results/Claim-Detection/similar_claims_kt.xlsx', engine='xlsxwriter')
for dType, group in average_percentages_df.groupby('Type'):
    group.drop('Type', axis=1).sort_values(by="Model", ascending=False).to_excel(excel_writer, sheet_name=dType,
                                                                                   index=False)
excel_writer.close()

"""
columns_to_mean = ["accuracy", "accuracy-std", "precision", "precision-std", "recall", "recall-std", "f1_score",
                       "f1_score-std"]

print("Total number of records: ", test_data.shape[0])
###########################################
# Coarse-grained NER Presence

cNER_data = test_data[test_data["cNER_Exists"]]
cNER_data["source"] = "cNER"
print("cNER data size: ", cNER_data.shape[0])

resultsC = Results.getLanguageLevelResults(cNER_data, models, columns_to_mean)

###########################################
# Fine-grained NER Presence

fNER_data = test_data[test_data["fNER_Exists"]]
fNER_data["source"] = "fNER"
print("fNER data size: ", fNER_data.shape[0])

resultsF = Results.getLanguageLevelResults(fNER_data, models, columns_to_mean)

##########################################
# Combine both results

results = pd.concat([resultsC, resultsF], ignore_index=True)

source_wise_results = results.groupby(['source', 'model'])[columns_to_mean].mean()
source_wise_results = source_wise_results.reset_index()

# Write to Excel file
excel_writer = pd.ExcelWriter(path + 'Results/Claim-Detection/entity_presence_analysis.xlsx',
                                  engine='xlsxwriter')
for source, group in source_wise_results.groupby('source'):
    group.drop('source', axis=1).sort_values(by="model", ascending=False).to_excel(excel_writer, sheet_name=source,
                                                                                       index=False)
excel_writer.close()
"""