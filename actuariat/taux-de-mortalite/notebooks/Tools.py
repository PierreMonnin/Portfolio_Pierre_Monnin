#Librairies nécéssaires aux fonctions :
import pandas as pd
import numpy as np
from IPython.display import display, Markdown
import networkx as nx
from numpy.linalg import eigvalsh
import matplotlib.pyplot as plt
#Préparation de data base :
 
## Fonction d'import / export : 

def import_csv(chemin, sep):
    data = pd.read_csv(chemin, sep=sep, encoding="latin1")
    # 🔹 Dimensions du dataset
    display(Markdown("### 💠 Dimensions du dataset"))
    display(data.shape)
     # 🔹 Affichage des premières lignes
    display(Markdown("### 💠 Visualisation des 5 premières lignes"))
    pd.set_option('display.max_columns', None)
    display(data.head(5))
    return data
#chemin = chemin du fichier à définir avec  chemin=Path("").resolve(), attention: séparateur pour emplacement dans ce sens / 
#sep = marqueur de séparation du fichier csv à définir directement dans l'appel de la fonction ""

def export_csv(data,chemin, sep):
    data.to_csv(chemin, index=False, sep=sep, encoding="latin1")
#data= base de données à exporter
#chemin= chemin de stockage de l'export
#sep= marqueur de séparation du fichier csv crée


## Fonction de jointure : 

def full_concatenate(liste,axis): 
    data = pd.concat(liste, axis=axis, ignore_index=True)
        # 🔹 Dimensions du dataset
    display(Markdown("### 💠 Dimensions du dataset"))
    display(data.shape)
     # 🔹 Affichage des premières lignes
    display(Markdown("### 💠 Visualisation des 5 premières lignes"))
    pd.set_option('display.max_columns', None)
    display(data.head(5))
    return data
#liste= défini la liste des data set à concaténer: [PSG,OM,OL,...]
#axis = concaténation en ligne à ligne = 0, colonne en colonne = 1

## Fonction de Manip data base : 

def suppression_colonnes(base,liste):
    data = base.drop(liste, axis=1)
            # 🔹 Dimensions du dataset
    display(Markdown("### 💠 Dimensions du dataset"))
    display(data.shape)
     # 🔹 Affichage des premières lignes
    display(Markdown("### 💠 Visualisation des 5 premières lignes"))
    pd.set_option('display.max_columns', None)
    display(data.head(5))
    return data
#base = dataset en output
#liste= liste des colonnes à supprimer (par leur nom)

def groupeby(Input,liste,Cible):
    Output = (
    Input
    .groupby(liste, as_index=False)[Cible]
    .sum()
    )
    # 🔹 Dimensions du dataset
    display(Markdown("### 💠 Dimensions du dataset"))
    display(Output.shape)
    # 🔹 Affichage des premières lignes
    display(Markdown("### 💠 Visualisation des 5 premières lignes"))
    pd.set_option('display.max_columns', None)
    display(Output.head(5))
    return Output
#Input= data base prise en entrée 
#liste= liste des variables à garder dans Output
#Cible= variable sur laquelle on fait le groupby 

def left_join(gauche,droite,cléETvar, clé):
    df_merge = gauche.merge(
        droite[cléETvar],   # on ne garde que les colonnes utiles dans droite (clé et var à ramener)
        on=clé,              # clés de jointure
        how="left"                  # "left" = garde toutes les lignes de gauche
    )
    # 🔹 Dimensions du dataset
    display(Markdown("### 💠 Dimensions du dataset"))
    display(df_merge.shape)
    # 🔹 Affichage des premières lignes
    display(Markdown("### 💠 Visualisation des 5 premières lignes"))
    pd.set_option('display.max_columns', None)
    display(df_merge.head(5))
    return df_merge
#gauche= data set sur lequel on fait la jointure 
#droite= data set que l'on joint
#cléETvar= clé de jointure puis variable que l'on veut ramener 
#clé = clé de jointure

def suppression_ligne_modalité(Data,Variable,Modalité):
    Data_out = Data[~Data[Variable].isin(Modalité)]
    #Check visuel de bon fonctionnement de la fonction: 
    nb_lignes_entrée = Data[Data[Variable].isin(Modalité)].shape[0]
    Check1= len(Data)-len(Data_out)-nb_lignes_entrée
    display(Markdown("### 💠 Premier Check :"))
    display(Check1)
    Check2 = Data_out[Data_out[Variable].isin(Modalité)].shape[0]
    display(Markdown("### 💠 Second Check :"))
    display(Check2)
    if Check1 ==0 and Check2==0:
        display(Markdown("### ✅ Modalités Supprimées"))
    else: 
        display(Markdown("### ⛔ Le traitement n'a pas fonctionné"))
    return Data_out
#Data = base de donnée dont on part
#Variable= Variable sur laquelle on choisit modalité à supprimer
#Modalité= modalité que l'on veut supprimer

def index_df(df, row_var, col_var, value_var):
    Tableau = df.pivot_table(index=row_var,
                          columns=col_var,
                          values=value_var,
                         )
    display(Markdown("### 💠 Dimensions du dataset"))
    display(Tableau.shape)
    display("💠 Affichage du tableau :")
    display(Tableau)
    return Tableau
#df = data set d'entree
#row_var = variable indexé en ligne
#col_var = variable indexé en colonne
#value_var = variable en valeur
 
#Dictionnaire : 

def dict_to_dataframe(d):
    out = {}
    for k, v in d.items():
        a = np.asarray(v)

        # Cas 0D (scalaire) -> on le répète si possible plus tard
        if a.ndim == 0:
            out[k] = a  # on stocke tel quel, on traitera après
            continue

        # Cas 1D -> ok
        if a.ndim == 1:
            out[k] = a.tolist()
            continue

        # Cas 2D (n,1) -> on aplati en 1D
        if a.ndim == 2 and a.shape[1] == 1:
            out[k] = a.ravel().tolist()
            continue

        # Cas 2D (n,m>1) -> on crée plusieurs colonnes k_0, k_1, ...
        if a.ndim == 2 and a.shape[1] > 1:
            for j in range(a.shape[1]):
                out[f"{k}_{j}"] = a[:, j].tolist()
            continue

        # Cas >=3D -> on aplati les deux dernières dims en colonnes
        if a.ndim >= 3:
            n = a.shape[0]
            a2 = a.reshape(n, -1)
            for j in range(a2.shape[1]):
                out[f"{k}_{j}"] = a2[:, j].tolist()
            continue

    # Harmonisation des longueurs + scalaires
    # On cherche la longueur max des listes
    lens = [len(v) for v in out.values() if isinstance(v, list)]
    n = max(lens) if lens else 1
    for k, v in list(out.items()):
        if not isinstance(v, list):           # scalaire
            out[k] = [v] * n
        elif len(v) != n:                     # on tronque / pad si besoin
            out[k] = (v + [None] * n)[:n]

    return pd.DataFrame(out)
    

#Fonction de check : 

def Check_variable_dataset(df):
    for col in df.columns:
        modalites = df[col].dropna().unique()
        display(Markdown("### "f"\n👓 {col} ({len(modalites)} modalités)"))
        display(modalites)
#Objectif: afficher les modalités variable par variable, permet de checker ou d'observer
#df= base de donnée que l'on explore ou que l'on check 

#Fonction graphe : 

def Construction_adjacence_by_voisinage(regions, voisin, weight=1.0):
    n = len(regions)
    W = np.zeros((n, n), dtype=float)
    idx = {r: i for i, r in enumerate(regions)}

    for r, neighs in voisin.items():
        if r not in idx:
            continue
        i = idx[r]
        for nb in neighs:
            if nb in idx:
                j = idx[nb]
                W[i, j] = weight
                W[j, i] = weight  # symétrique (graphe non orienté), hypothèse théorique donnée dans le rapport

    return pd.DataFrame(W, index=regions, columns=regions)

def Calcul_du_Laplacien(W_matrix, normalized=True):
    W = W_matrix.values
    d = W.sum(axis=1)
    D = np.diag(d)
    L = D - W
    if not normalized:
        return pd.DataFrame(L, index=W_matrix.index, columns=W_matrix.columns)
    # L_sym = I - D^{-1/2} W D^{-1/2}
    with np.errstate(divide="ignore"):
        d_sqrt_inv = np.diag(1.0 / np.sqrt(np.where(d > 0, d, 1)))
    L_sym = np.eye(W.shape[0]) - d_sqrt_inv @ W @ d_sqrt_inv
    return pd.DataFrame(L_sym, index=W_matrix.index, columns=W_matrix.columns)
#W_matrix= matrice d'adjacence de notre graphe
#normalized= True, renvoie le Laplacien normalisé, False, renvoie le Laplacien de base


def check_laplacien(L, W, atol=1e-10, name="L"):
    L = np.asarray(L, dtype=float)
    W = np.asarray(W, dtype=float)

    display(f"== Vérification {name} ==")

    # 1. Symétrie
    if np.allclose(L, L.T, atol=atol):
        display("✅ Symétrique")
    else:
        display("❌ Non symétrique")

    # 2. Structure diag / hors diag
    d = W.sum(axis=1)
    diag_ok = np.allclose(np.diag(L), d, atol=atol)
    off_diag_ok = np.allclose(L - np.diag(np.diag(L)), -W + np.diag(np.diag(W)), atol=atol)
    if diag_ok and off_diag_ok:
        display("✅ Structure correcte (diag=degrés, hors diag=-w_ij)")
    else:
        display("❌ Structure incorrecte")

    # 3. Somme des lignes
    row_sums = L.sum(axis=1)
    if np.allclose(row_sums, 0.0, atol=atol):
        display("✅ Somme des lignes = 0")
    else:
        display("❌ Somme des lignes non nulles")
        display("Exemple :", row_sums[:5])

    # 4. Spectre
    valeurs_propres = np.linalg.eigvalsh(L)
    if (valeurs_propres >= -atol).all():
        display("✅ Valeurs propres ≥ 0")
    else:
        display("❌ Valeurs propres négatives détectées")
    if valeurs_propres.min()<=atol:
        display("λ min =", 0, "λ max =", round(valeurs_propres.max(),2))
    else:
        display("❌ 0 n'est pas valeur propre")

    # 5. Multiplicité de λ=0
    mult_zero = np.sum(np.isclose(valeurs_propres, 0.0, atol=atol))

    # nb composantes connexes du graphe
    G = nx.from_numpy_array(W)
    nb_cc = nx.number_connected_components(G)

    if mult_zero == nb_cc:
        display(f"✅ Multiplicité de λ=0 = {mult_zero}, correspond aux {nb_cc} composantes connexes")
    else:
        display(f"❌ Multiplicité de λ=0 ({mult_zero}) ≠ nb composantes ({nb_cc})")

    return valeurs_propres
#L = Matrice du Laplacien à vérifier
#W = Matrice d'adjacence
#name= nom du Laplacien
#atol=seuil de tolérence à 0 

def check_laplacien_normalise(L, W, atol=1e-10, name="L"):
    display(f"== Vérification {name} ==")
    L = np.asarray(L, dtype=float)
    W = np.asarray(W, dtype=float)
    d = W.sum(axis=1)

    # Spectre
    valeurs_propres = eigvalsh(L)
    if (valeurs_propres >= -atol).all() and (valeurs_propres <= 2 + atol).all():
        display("✅ Spectre contenu dans [0,2]")
    else:
        display("❌ Spectre hors [0,2]")
        display("λ min =", valeurs_propres.min(), "λ max =", valeurs_propres.max())

    # Test RW vs Sym
    row_sums = L.sum(axis=1)

    if np.allclose(row_sums, 0.0, atol=1e-8):
        display("👉 Détection : Laplacien random-walk")
        display("✅ Somme des lignes = 0")
    else:
        display("👉 Détection : Laplacien symétrique normalisé")
        v = np.sqrt(d)
        if np.allclose(L @ v, 0.0, atol=1e-8):
            display("✅ Condition L * sqrt(d) ≈ 0 respectée")
        else:
            display("❌ Condition L * sqrt(d) non respectée")
            display("Norme résiduelle :", np.linalg.norm(L @ v))

    return valeurs_propres
#L = Matrice du Laplacien à vérifier
#W = Matrice d'adjacence
#name= nom du Laplacien
#atol=seuil de tolérence à 0 

## Fonction graphique :
def plot_on_graph(G, valeurs, titre="", vmin=None, vmax=None, cmap="viridis"):
    """
    Trace les arêtes à partir de G.W puis colore les nœuds par 'valeurs'.
    Compatible toutes versions PyGSP (pas d'arguments exotiques).
    """
    assert len(valeurs) == G.N, "Taille de 'valeurs' != nombre de nœuds."
    if not hasattr(G, "coords") or G.coords is None:
        raise ValueError("G n'a pas de coordonnées. Fais G.set_coordinates(coords_array) avant.")

    # bornes couleurs
    vmin = np.min(valeurs) if vmin is None else vmin
    vmax = np.max(valeurs) if vmax is None else vmax

    # coords
    XY = np.asarray(G.coords)
    x, y = XY[:, 0], XY[:, 1]

    # --- figure ---
    plt.figure(figsize=(6, 5))

    # 1 tracer les arêtes depuis la matrice d'adjacence (sparse)
    W = G.W.tocoo() if hasattr(G.W, "tocoo") else np.asarray(G.W)
    if hasattr(W, "row"):  # sparse
        for i, j, w in zip(W.row, W.col, W.data):
            if i < j and w != 0:  # éviter doublons
                plt.plot([x[i], x[j]], [y[i], y[j]], color="0.75", lw=0.8, zorder=1)
    else:  # dense
        n = W.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                if W[i, j] != 0:
                    plt.plot([x[i], x[j]], [y[i], y[j]], color="0.75", lw=0.8, zorder=1)

    # 2) scatter des nœuds colorés
    sc = plt.scatter(x, y, c=valeurs, s=120, cmap=cmap, vmin=vmin, vmax=vmax, zorder=3)

    plt.colorbar(sc, label="Coefficient SGWT")
    plt.title(titre)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


