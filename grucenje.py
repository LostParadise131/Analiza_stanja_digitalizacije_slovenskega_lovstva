import os
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from maps_for_questions import (
    answer_map,
    indexes_for_single_choice_questions,
    question_map,
)

# --- For pandas DataFrames ---
pd.set_option("display.max_rows", None)  # show all rows
pd.set_option("display.max_columns", None)  # show all columns
pd.set_option("display.width", None)  # auto-detect width
pd.set_option("display.max_colwidth", None)  # show full content in each column
pd.set_option("display.float_format", "{:.4f}".format)  # format floats nicely

# --- For NumPy arrays ---
np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=4)
os.makedirs("slike_od_grucenja", exist_ok=True)

# Open a text file to save all prints
with open("output_iz_grucenje_py.txt", "w", encoding="utf-8") as f:
    with redirect_stdout(f):

        # Step 1: Naloži podatke ter odstrani neuporabne stolpce
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
                data[col] = data[col].map(
                    {1: 0, 2: (1 / 4), 3: (2 / 4), 4: (3 / 4), 5: 1}
                )
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
        dist_matrix = pdist(
            data_scaled, metric="cityblock"
        )  # condensed distance matrix

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
        plt.savefig(
            "slike_od_grucenja/silhuetna_analiza.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        # Step 7: t-SNE embedding fou visualization
        tsne = TSNE(n_components=2, perplexity=30, metric="cityblock", random_state=42)
        embedding = tsne.fit_transform(data_scaled)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=cluster_labels,
            cmap="tab10",
            s=50,
            alpha=0.8,
        )
        plt.title("Vizualizacija podatkov z uporabo t-SNE")
        plt.xlabel("t-SNE dimenzija 1")
        plt.ylabel("t-SNE dimenzija 2")

        # legend by cluster
        plt.legend(*scatter.legend_elements(), title="Gruče", loc="best")

        plt.savefig(
            "slike_od_grucenja/tsne_vizualizacija.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        # Step 8: PCA to analyze variance contribution
        pca = PCA()
        pca.fit(data_scaled)

        # Explained variance ratio per component
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Plot cumulative variance
        plt.figure(figsize=(8, 5))
        plt.plot(
            range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o"
        )
        plt.axhline(y=0.9, color="r", linestyle="--", label="90% variance")
        plt.xlabel("Število glavnih komponent (PC)")
        plt.ylabel("Kumulativni delež variance")
        plt.title("PCA: Kumulativna razlaga variance")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            "slike_od_grucenja/pca_cumulative_variance.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # Doloci koliko komponent potrebujemo za 90% variance
        num_components_90 = np.argmax(cumulative_variance >= 0.9) + 1
        print(f"Število glavnih komponent za 90% variance: {num_components_90}")

        loadings = pca.components_.T  # shape: features x components

        # 3 `num_components_90`
        question_texts = [question_map.get(col, col) for col in column_list]
        feature_contributions = np.sum(np.abs(loadings[:, :num_components_90]), axis=1)
        feature_importance = pd.DataFrame(
            {
                "Oznaka": column_list,
                "Vprasanje": question_texts,
                "Contribution": feature_contributions,
            }
        ).sort_values(by="Contribution", ascending=False)

        print("Vpliv posameznih vprašanj na 90% variance:")
        for idx, row in feature_importance.iterrows():
            print(f"{row['Oznaka']} {row['Vprasanje']}:\n {row['Contribution']:.4f}")

        # Step 9: Compute cluster profiles
        cluster_labels = fcluster(Z, best_k, criterion="maxclust")
        data_profiles = data.copy()
        data_profiles["Cluster"] = cluster_labels

        cluster_profile_avg = data_profiles.groupby("Cluster").mean()
        cluster_profile_avg.columns = [
            question_map.get(col, col) for col in cluster_profile_avg.columns
        ]

        print("Povprečne vrednosti vprašanj po gruče:")
        # Loop through questions first, then print each cluster's median
        for question in cluster_profile_avg.columns:
            print(f"{question}")
            for cluster_id in cluster_profile_avg.index:
                value = cluster_profile_avg.loc[cluster_id, question]
                print(f"Gruča {cluster_id}: {value:.4f}")
            print()  # empty line between questions

        # Step 9a: Heatmap visualization
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            cluster_profile_avg,
            annot=True,
            cmap="YlGnBu",
            cbar_kws={"label": "Povprečna vrednost"},
        )
        plt.title(f"Profil gruč za {best_k} gruče")
        plt.xlabel("Vprašanja")
        plt.ylabel("Gruče")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("slike_od_grucenja/cluster_profiles_heatmap.png", dpi=300)
        plt.show()
