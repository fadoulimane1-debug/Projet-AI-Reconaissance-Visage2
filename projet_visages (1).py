# ============================================================ 
# PROJET : Reconnaissance de visages avec LPQ + KNN + PCA
# ============================================================
# AVANT D'EXECUTER : dans le terminal tape :
# pip install scikit-learn numpy matplotlib scikit-image scipy

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer

# ============================================================
# ETAPE 1 : Charger le Dataset LFW
# ============================================================
print("=" * 50)
print("ETAPE 1 : Chargement du dataset LFW...")
print("=" * 50)

lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw.images
y = lfw.target
target_names = lfw.target_names

print(f"Nombre d'images     : {X.shape[0]}")
print(f"Taille image        : {X.shape[1]}x{X.shape[2]}")
print(f"Nombre de personnes : {len(target_names)}")
print(f"Personnes           : {list(target_names)}")

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle("ETAPE 1 : Exemples du Dataset LFW", fontsize=14, fontweight='bold')
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X[i], cmap='gray')
    ax.set_title(target_names[y[i]], fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.savefig("etape1_dataset.png")
plt.show()
print("-> Image sauvegardee : etape1_dataset.png\n")

# ============================================================
# ETAPE 2 : Fonction LPQ + Extraction des features
# ============================================================
print("=" * 50)
print("ETAPE 2 : Extraction des features LPQ...")
print("=" * 50)

def lpq(image, winSize=3):
    image = image.astype(float)
    w0 = np.ones((1, winSize))
    w1 = np.exp(-2j * np.pi * np.arange(winSize) / winSize).reshape(1, -1)
    w2 = w1 ** 2
    w3 = w1 ** 3
    filters = [np.dot(f1.T, f2) for f1 in [w0, w1, w2, w3]
                                 for f2 in [w0, w1, w2, w3]]
    responses = []
    for f in filters:
        resp = convolve2d(image, np.conj(f), mode='valid')
        responses.append(resp.real)
        responses.append(resp.imag)
    lpq_code = np.zeros_like(responses[0], dtype=int)
    for i, r in enumerate(responses):
        lpq_code += (r >= 0).astype(int) * (2 ** i)
    hist, _ = np.histogram(lpq_code.ravel(), bins=256, range=(0, 256))
    total = hist.sum()
    if total == 0:
        return np.zeros(256)
    return hist / total

X_lpq = np.array([lpq(img) for img in X])

# CORRECTION NaN
print(f"NaN detectes avant nettoyage : {np.isnan(X_lpq).sum()}")
imputer = SimpleImputer(strategy='constant', fill_value=0)
X_lpq = imputer.fit_transform(X_lpq)
print(f"NaN apres nettoyage : {np.isnan(X_lpq).sum()}")
print(f"Features LPQ extraites : {X_lpq.shape}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("ETAPE 2 : Descripteur LPQ", fontsize=14, fontweight='bold')
axes[0].imshow(X[0], cmap='gray')
axes[0].set_title(f"Image : {target_names[y[0]]}")
axes[0].axis('off')
axes[1].bar(range(256), X_lpq[0], color='steelblue', width=1)
axes[1].set_title("Histogramme LPQ")
axes[1].set_xlabel("Bins")
axes[1].set_ylabel("Frequence normalisee")
plt.tight_layout()
plt.savefig("etape2_lpq.png")
plt.show()
print("-> Image sauvegardee : etape2_lpq.png\n")

# ============================================================
# ETAPE 3 : Division Train / Test
# ============================================================
print("=" * 50)
print("ETAPE 3 : Division Train / Test...")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X_lpq, y, test_size=0.25, random_state=42
)
print(f"Train : {X_train.shape[0]} images")
print(f"Test  : {X_test.shape[0]} images")

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("ETAPE 3 : Repartition Train / Test", fontsize=14, fontweight='bold')
ax.bar(['Train', 'Test'], [X_train.shape[0], X_test.shape[0]],
       color=['steelblue', 'orange'], width=0.4)
ax.set_ylabel("Nombre d'images")
for i, v in enumerate([X_train.shape[0], X_test.shape[0]]):
    ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("etape3_split.png")
plt.show()
print("-> Image sauvegardee : etape3_split.png\n")

# ============================================================
# ETAPE 4 : PCA
# ============================================================
print("=" * 50)
print("ETAPE 4 : Application de PCA...")
print("=" * 50)

pca = PCA(n_components=100, whiten=True)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"Avant PCA : {X_train.shape[1]} dimensions")
print(f"Apres PCA : {X_train_pca.shape[1]} dimensions")

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("ETAPE 4 : Variance expliquee par PCA", fontsize=14, fontweight='bold')
ax.plot(np.cumsum(pca.explained_variance_ratio_) * 100, color='green', linewidth=2)
ax.set_xlabel("Nombre de composantes")
ax.set_ylabel("Variance cumulee (%)")
ax.axhline(y=90, color='red', linestyle='--', label='90%')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("etape4_pca.png")
plt.show()
print("-> Image sauvegardee : etape4_pca.png\n")

# ============================================================
# ETAPE 5 : KNN
# ============================================================
print("=" * 50)
print("ETAPE 5 : Classification KNN...")
print("=" * 50)

k_values = [1, 3, 5, 7, 9, 11]
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_pca, y_train)
    y_pred = knn.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"  k={k:2d}  -->  Accuracy = {acc*100:.2f}%")

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("ETAPE 5 : Accuracy KNN selon k", fontsize=14, fontweight='bold')
ax.plot(k_values, [a * 100 for a in accuracies],
        marker='o', color='blue', linewidth=2, markersize=8)
for k, a in zip(k_values, accuracies):
    ax.annotate(f"{a*100:.1f}%", (k, a*100),
                textcoords="offset points", xytext=(0, 10), ha='center')
ax.set_xlabel("Valeur de k")
ax.set_ylabel("Accuracy (%)")
ax.set_xticks(k_values)
ax.grid(True)
plt.tight_layout()
plt.savefig("etape5_accuracy_k.png")
plt.show()
print("-> Image sauvegardee : etape5_accuracy_k.png\n")

# ============================================================
# ETAPE 6 : Matrice de confusion
# ============================================================
print("=" * 50)
print("ETAPE 6 : Matrice de confusion...")
print("=" * 50)

best_k = k_values[np.argmax(accuracies)]
print(f"Meilleur k = {best_k}  |  Accuracy = {max(accuracies)*100:.2f}%")

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_pca, y_train)
y_pred_best = knn_best.predict(X_test_pca)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred_best, target_names=target_names))

fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle(f"ETAPE 6 : Matrice de confusion (k={best_k})",
             fontsize=14, fontweight='bold')
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best,
    display_labels=target_names,
    xticks_rotation=45,
    ax=ax,
    colorbar=False
)
plt.tight_layout()
plt.savefig("etape6_confusion.png")
plt.show()
print("-> Image sauvegardee : etape6_confusion.png\n")

# ============================================================
# ETAPE 7 : Visualisation predictions
# ============================================================
print("=" * 50)
print("ETAPE 7 : Visualisation des predictions...")
print("=" * 50)

_, X_test_images, _, _ = train_test_split(
    lfw.images, y, test_size=0.25, random_state=42
)

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("ETAPE 7 : Predictions vs Verite", fontsize=14, fontweight='bold')
for i, ax in enumerate(axes.ravel()):
    if i < len(X_test_images):
        ax.imshow(X_test_images[i], cmap='gray')
        pred = target_names[y_pred_best[i]]
        true = target_names[y_test[i]]
        color = 'green' if pred == true else 'red'
        ax.set_title(f"Pred: {pred}\nVrai: {true}", fontsize=7, color=color)
        ax.axis('off')
plt.tight_layout()
plt.savefig("etape7_predictions.png")
plt.show()
print("-> Image sauvegardee : etape7_predictions.png\n")

print("=" * 50)
print("PROJET TERMINE ! Toutes les images ont ete sauvegardees.")
print("=" * 50)
