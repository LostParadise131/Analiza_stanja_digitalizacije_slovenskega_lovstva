import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from maps_for_questions import (
    answer_map,
    indexes_for_single_choice_questions,
    question_map,
)
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

# Make sure folder exists
os.makedirs("slike_od_grucenja", exist_ok=True)
# Step 1: Load data and exclude non relevant questions/columns
data = pd.read_excel("podatki.xlsx")
# column_list = data.columns.to_list()
# print(f"col_list:{column_list}")
# print(f"index_single_choice:{indexes_for_single_choice_questions}")
# print("Q2, Q3, Q4, Q5, Q7, Q8, Q10, Q11, Q12, Q13, Q14 ")
data = data.iloc[1:, indexes_for_single_choice_questions]
column_list = data.columns.to_list()
# print(f"col_list:{column_list}")

# Step 2: Deal with missing values by imputing the column mean
# from 1 to 0, 1 meaning more tech savvy and 0 less tech savvy
for col in data.columns:
    if col == "Q2":
        data[col] = data[col].map({1: 1, 2: 0, 3: 0.75})
    elif col == "Q3":
        data[col] = data[col].map(
            {1: 1, 2: (5 / 6), 3: (4 / 6), 4: (1 / 2), 5: (2 / 6), 6: 0}
        )
    elif col == "Q4":
        data[col] = data[col].map({1: 0, 2: (1 / 4), 3: (2 / 4), 4: (3 / 4), 5: 1})
    elif col == "Q5":
        data[col] = data[col].map({1: 1, 2: 0})
    elif col == "Q7":
        data[col] = data[col].map({1: 1, 2: 0})
    elif col == "Q8":
        data[col] = data[col].map({1: 1, 2: 0})
    elif col == "Q10":
        data[col] = data[col].map({1: 1, 2: (2 / 3), 3: (1 / 3), 4: 0})
    elif col == "Q11":
        data[col] = data[col].map({1: 1, 2: 0})
    elif col == "Q12":
        data[col] = data[col].map({1: 1, 2: (2 / 3), 3: (1 / 3), 4: 0})
    elif col == "Q13":
        data[col] = data[col].map({1: 1, 2: (2 / 3), 3: (1 / 3), 4: 0})
    elif col == "Q14":
        data[col] = data[col].map({1: 1, 2: (2 / 3), 3: (1 / 3), 4: 0})

data = data.replace(-2, np.nan)  # replace -2 with NaN
data = data.fillna(data.mean())  # impute NaN with column mean

print(f"num_rows:{data.shape[0]}, num_cols:{data.shape[1]}")

# Step 3: Normalize (0-1 scaling)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Step 3: Compute Manhattan distance
dist_matrix = pdist(data_scaled, metric="cityblock")  # condensed distance matrix

# Step 4: Hierarchical clustering (average linkage)
Z = linkage(dist_matrix, method="average")

# Step 5: Determine optimal k using silhouette score
sil_scores = {}
for k in range(2, 15):
    cluster_labels = fcluster(Z, k, criterion="maxclust")
    score = silhouette_score(
        squareform(dist_matrix), cluster_labels, metric="precomputed"
    )
    sil_scores[k] = score

best_k = max(sil_scores, key=sil_scores.get)
print("Najboljše število gruč", best_k, "s silhueto", sil_scores[best_k])

# Step 6a: Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z, truncate_mode="lastp", p=20, show_leaf_counts=True)
plt.axhline(y=Z[-best_k, 2], color="r", linestyle="--")  # optional cutoff line
plt.title("Dendrogram hierarhičnega gručevanja")
plt.xlabel("Indeksi vzorcev ali (velikost gruče)")
plt.ylabel("Razdalja")
plt.savefig("slike_od_grucenja/dendrogram.png", dpi=300, bbox_inches="tight")
plt.show()

# Step 6b: Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(list(sil_scores.keys()), list(sil_scores.values()), marker="o")
plt.xlabel("Število gruč (k)")
plt.ylabel("Povprečna vrednost silhuete")
plt.title("Silhuetna analiza za hierarhično gručevanje")
plt.savefig("slike_od_grucenja/silhuetna_analiza.png", dpi=300, bbox_inches="tight")
plt.show()
