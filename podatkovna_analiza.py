import os
import sys
from contextlib import redirect_stdout

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from maps_for_questions import (
    answer_map,
    indexes_for_single_choice_questions,
    question_map,
)

CUTOFF = 50  # za dolga vprasanja

# Diplay settings
pd.set_option("display.max_rows", None)  # show all rows
pd.set_option("display.max_columns", None)  # show all columns
pd.set_option("display.width", None)  # auto-detect width
pd.set_option("display.max_colwidth", None)  # show full content in each column
pd.set_option("display.float_format", "{:.4f}".format)  # format floats nicely

np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=4)
os.makedirs("slike", exist_ok=True)

# Put all prints in print_output.txt
with open("print_output.txt", "w", encoding="utf-8") as f:
    with redirect_stdout(f):
        # Step 1: Nalozi podatke ter odstrani neuporabne stolpce
        data = pd.read_excel("podatki.xlsx")

        # Odstanimo se Q8 vprasanje, ker je vec kot 75% NaN
        indexes_for_single_choice_questions.remove(9)
        # magic numbers ik :(

        data = data.iloc[1:, indexes_for_single_choice_questions]
        column_list = data.columns.to_list()
        question_texts = [question_map[col] for col in column_list]

        # Step 2: Zamenjaj od 1KA NaN z np.nan
        data = data.replace(-2, np.nan)
        data = data.replace(-3, np.nan)

        # Step 3: Omejuj vrednosti na [0,1]
        # 1 pomeni bolj digitalno angaziran, 0 manj
        # vrednosti so izbrane glede na smiselnost vprasanj
        # POGLEJ question_map in answer_map
        for col in data.columns:
            if col == "Q2":
                data[col] = data[col].map({1: 1, 2: 0, 3: 0.75})
            elif col == "Q3":
                data[col] = data[col].map(
                    {
                        1: 1,
                        2: (5 / 6),
                        3: (4 / 6),
                        4: (1 / 2),
                        5: (2 / 6),
                        6: 0,
                        np.nan: 0,  # ljudlje ki niso odgovorili na to vprasanje nimajo spletne strani
                    }
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

        # prikazi mean in NaN delež stolpcov za vsak stolpec
        print("OSNOVNI PODATKI:")
        print(f"st_vrst:{data.shape[0]}, st_stolpcev:{data.shape[1]}")
        print("Mean in stevilo NaN za vsak stolpec:")
        for col in data.columns:
            mean_val = data[col].mean()
            median_val = data[col].median()
            print(
                f"{question_map[col]}\n"
                + f"Mean = {mean_val:.4f}, "
                + f"Stevilo NaN = {(data[col].isna().sum()):.4f}\n"
            )
        data = data.fillna(data.mean())

        # Step 4: Izracunaj razdaljo med vrsticami nato pa hierarchicno grucenje
        print("\nHIERARHICNO GRUCENJE:")
        distance_types = ["cityblock", "euclidean"]
        for dist_type in distance_types:
            dist_matrix = pdist(data, metric=dist_type)

            if dist_type == "euclidean":
                Z = linkage(dist_matrix, method="ward")
            else:
                Z = linkage(dist_matrix, method="complete")

            # Step 5: determiniraj "optimalno" stevilo gruc
            sil_scores = {}
            for k in range(2, 31):
                cluster_labels = fcluster(Z, k, criterion="maxclust")
                score = silhouette_score(
                    squareform(dist_matrix), cluster_labels, metric="precomputed"
                )
                sil_scores[k] = score

            best_k = max(sil_scores, key=sil_scores.get)
            if dist_type == "cityblock":
                dist_type = "manhattanska razdalja"
            else:
                dist_type = "evklidska razdalja"

            print(
                "\n'Najboljse' stevilo gruc",
                best_k,
                "s silhueto",
                sil_scores[best_k],
                "z razdaljo",
                dist_type,
            )

            # Step 6a: Plot dendrogram
            plt.figure(figsize=(10, 6))
            dendrogram(Z, truncate_mode="lastp", p=20, show_leaf_counts=True)
            plt.axhline(y=Z[-best_k, 2], color="r", linestyle="--")
            plt.title("Dendrogram hierarhicnega grucenja (" + dist_type + ")")
            plt.xlabel("Indeksi vzorcev ali (velikost gruce)")
            plt.ylabel("Razdalja")
            plt.savefig(
                "slike/dendrogram_" + dist_type + ".png", dpi=300, bbox_inches="tight"
            )
            plt.show()

            # Step 6b: Plot silhouette scores
            plt.figure(figsize=(10, 6))
            plt.plot(list(sil_scores.keys()), list(sil_scores.values()), marker="o")
            plt.xlabel("Stevilo gruč (k)")
            plt.ylabel("Povprecna vrednost silhuete")
            plt.title("Silhuetna analiza za hierarhicno grucenja (" + dist_type + ")")
            plt.savefig(
                "slike/silhuetna_analiza" + dist_type + ".png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()

        # Naredi zanimive/razlozljive gruce
        # od 2 do 4 bom raziskal, ker so podatki binarni
        # ("ali podpiras _ in _ ali ne" je vecino vprasanj)

        # ODLOCIL ZA MANHATANSKO RAZDALJO, ker pri manjsemu stevilu gruc
        # (2-4) je silhueta boljsa TER VIZUALNO na dendrogramu izgleda bolj smiselno
        # GLEJ DENDROGRAM v .slike/

        # distance_types = ["cityblock", "euclidean"]
        # for dist_type in distance_types:

        print("\n\nRAZLOZLJIVE GRUCE:")

        dist_matrix = pdist(data, metric="cityblock")

        Z = linkage(dist_matrix, method="complete")

        sil_scores = {}
        for k in range(2, 5):
            cluster_labels = fcluster(Z, k, criterion="maxclust")
            score = silhouette_score(
                squareform(dist_matrix), cluster_labels, metric="precomputed"
            )
            sil_scores[k] = score
            # tabela vrstice: gruce, stolpci: povprecne vrednosti vprasanj
            data_with_clusters = data.copy()
            data_with_clusters["Gruca"] = cluster_labels
            df_cluster_means = data_with_clusters.groupby("Gruca").mean().round(4)
            if k == 3:
                print("\n------------------------------------------------------------")
                print(f"IZBRANA tabela podatkov za {k} gruc:")
                print(df_cluster_means)
                print("------------------------------------------------------------\n")
            else:
                print(f"Tabela podatkov za {k} gruc:")
                print(df_cluster_means)

        # Step 6a: Plot dendrogram
        plt.figure(figsize=(10, 6))
        dendrogram(Z, truncate_mode="lastp", p=20, show_leaf_counts=True)
        plt.title(
            "Dendrogram hierarhicnega grucenja ("
            + "manhatanska razdalja"
            + "),\n"
            + "silhuete: "
            + ", ".join([f"{k}:{sil_scores[k]:.2f}" for k in sil_scores])
        )
        plt.xlabel("Indeksi vzorcev ali (velikost gruce)")
        plt.ylabel("Razdalja")
        plt.savefig(
            "slike/dendrogram_razlozljivo_manhatanska.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # IZBRAL SEM SI 3 GRUCE,
        # saj sta pri 4 grucah dve (2 in 4) skupini prevec enaki
        # pri 2 pa vidimo samo binarni konstrast med
        # "digitalno pismeni" in "digitalno nepismeni"
        # MEDTEM KO NAM 3 grupe prikazejo nek GRADIENT
        # med digitalno angaziranimi, povprecno angaziranimi
        # ter med neangaziranimi

        # Dodaj dataframe atribut gruce_index
        najzanimivejsi_k = 3
        cluster_labels = fcluster(Z, najzanimivejsi_k, criterion="maxclust")
        gruce_ime_map = {
            1: "Digitalno angazirani",
            2: "Neangazirani",
            3: "Povprecno angazirani",
        }
        data["gruce_index"] = cluster_labels
        data["gruce_ime"] = data["gruce_index"].map(gruce_ime_map)

        # supervized learning
        # decision tree za razlago gruc
        X = data.drop(columns=["gruce_index", "gruce_ime"])
        y = data["gruce_index"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        decision_tree = DecisionTreeClassifier(random_state=42)
        param_grid = {
            "max_depth": [2, 3, 4, 10],
            "min_samples_split": [4, 6, 12],
            "min_samples_leaf": [3, 5, 10],
        }
        grid_search = GridSearchCV(decision_tree, param_grid, cv=3, scoring="accuracy")
        grid_search.fit(X_train, y_train)

        best_tree = grid_search.best_estimator_
        best_tree.fit(X_train, y_train)
        train_score = best_tree.score(X_train, y_train)
        test_score = best_tree.score(X_test, y_test)

        plt.figure(figsize=(14, 9))
        ax = plt.gca()
        plot_tree(
            best_tree,
            feature_names=X.columns,
            class_names=[gruce_ime_map[i] for i in sorted(y.unique())],
            filled=True,
            rounded=True,
            fontsize=14,
            impurity=False,  # hides Gini
            label="all",  # shows class name in leaf
            proportion=False,  # avoids showing raw samples
        )
        for text_obj in ax.texts:
            txt = text_obj.get_text()
            # Example of text in node: "feature_name <= 0.75\nsamples = 42\nclass = X"
            lines = txt.split("\n")
            new_lines = []
            for line in lines:
                if line.startswith("samples"):
                    continue  # skip sample counts
                else:
                    new_lines.append(line)
            text_obj.set_text("\n".join(new_lines))

        plt.title(
            f"Odlocilno drevo z naj parametrom:{grid_search.best_params_}\n"
            + f"Rezultat učne/testne množice:{train_score:.4f}/{test_score:.4f}\n"
        )
        plt.tight_layout()
        plt.savefig(
            "slike/odlocilno_drevo_razlozljivo.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # RANDOM FOREST za feature importance
        random_forest = RandomForestClassifier(random_state=42)

        param_grid = {
            "n_estimators": [50, 100, 200],  # number of trees
            "max_depth": [2, 3, 4, 5],  # max depth of trees
            "min_samples_split": [2, 4],  # min samples to split
            "min_samples_leaf": [3, 5, 10],  # min samples in a leaf
        }

        grid_search_random_forest = GridSearchCV(
            random_forest, param_grid, cv=3, scoring="accuracy"
        )
        grid_search_random_forest.fit(X_train, y_train)

        best_random_forest = grid_search_random_forest.best_estimator_

        train_score = best_random_forest.score(X_train, y_train)
        test_score = best_random_forest.score(X_test, y_test)

        # Uporabim MDA namesto MDI, ker je MDI lahko pristranski
        # do spremenljivk z vec kategorijami
        # omejitve, kar se racunanja tice
        perm_importance = permutation_importance(
            best_random_forest,
            X_test,
            y_test,
            n_repeats=30,  # Number of shuffles
            random_state=42,
            scoring="accuracy",
        )

        feature_names = X.columns
        importance_df_mda = pd.DataFrame(
            {"Feature": feature_names, "Importance": perm_importance.importances_mean}
        ).sort_values(by="Importance", ascending=False)

        print("\n\nRANDOM FOREST - POMEMBNOST ZNACILK (MDA):")
        # Pokazi vse vrednosti za pomembnost znacilk
        print("Permutation Feature Importance (MDA) for all features:\n")
        for idx, row in importance_df_mda.iterrows():
            print(f"{row['Feature']:<25} Importance: {row['Importance']:.4f}")

        plt.figure(figsize=(10, 6))
        plt.barh(
            importance_df_mda["Feature"],
            importance_df_mda["Importance"],
            color="skyblue",
        )
        plt.xlabel("Permutation Feature Importance (MDA)")
        plt.title(
            "Naključni gozd, ocena pomembnosti značilk s permutacijami\n"
            + f"Train/test score: {train_score:.4f}/{test_score:.4f}"
        )
        plt.gca().invert_yaxis()  # Highest importance on top
        plt.tight_layout()
        plt.savefig(
            "slike/random_forest_feature_importance.png", dpi=300, bbox_inches="tight"
        )
        plt.show()
