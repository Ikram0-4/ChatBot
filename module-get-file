import sys
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append("/Users/ikramjanati/Documents/BDD_Pourvois") 

from sauvergarde_modul import get_files

files = get_files("../data")  

print(f"{len(files)} fichiers trouvés :")
for i, path in enumerate(files, start=1):
    print(f"{i}. {path}")

documents = []
for path in files:
    with open(path, "r", encoding="utf-8") as f:
        documents.append(f.read())

df = pd.DataFrame(documents, columns=["texte"])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["texte"])

print("\n Vectorisation terminée.")
print("Taille du vecteur :", X.shape)
