import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import scipy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
from nltk.corpus import stopwords
import mlflow
import mlflow.sklearn

import nltk
nltk.download('stopwords')

def load_imdb_reviews(ruta_archivo):
    df = pd.read_csv(ruta_archivo)
    return df

df_train = load_imdb_reviews('./imdb_reviews_train.csv')
print("\nPrimeras 5 filas del conjunto de datos de entrenamiento de reseñas de IMDB en en español:")
df_train.head()

df_test = load_imdb_reviews('./imdb_reviews_test.csv')
print("\nPrimeras 5 filas del conjunto de datos de prueba de reseñas de IMDB en en español:")
df_test.head()

print(df_train.iloc[18].review)
print(df_train.iloc[18].sentiment)

def mostrar_distribucion_de_clases(df, stage):
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x='sentiment', data=df, palette=['#FF9999','#99FF99'])
    ax.set_xticklabels(['Negativa', 'Positiva'])
    if stage == 'train':
        plt.title('Distribución de reseñas del conjunto de entrenamiento por clase de sentimiento')
    else:
        plt.title('Distribución de reseñas del conjunto de pruebas por clase de sentimiento')
    plt.xlabel('Clase de Sentimiento')
    plt.ylabel('Cantidad de Reseñas')
    # plt.show()

mostrar_distribucion_de_clases(df_train, 'train')
mostrar_distribucion_de_clases(df_test, 'test')

for review in np.array(df_train)[np.array(df_train.sentiment) == 0][:5]:
    print(review)

for review in np.array(df_train)[np.array(df_train.sentiment) == 1][:5]:
    print(review)

def tokenize_text(text: str) -> "list[str]":
    """
    Procesa un texto y luego lo tokeniza.
    
    Args:
        text (str): Texto a procesar y tokenizar.
    
    Returns:
        list[str]: Lista de tokens.
    """
    text = re.sub(r'[^a-z]', ' ', text.lower())
    text = re.sub(r'\s[a-z]([a-z])?\b', '', text)
    return text.split()

sample = df_train.iloc[0].review
print('Texto original:')
print(sample)
print('Texto procesado y tokenizado:')
print(tokenize_text(sample))

vocabulary = {}
for r in np.array(df_train):
    for token in tokenize_text(r[0]):
        vocabulary[token] = vocabulary.get(token, 0) + 1

print(f'Número de tokens únicos: {len(vocabulary)}')
df_train.iloc[0]['review']

stop_words = set(stopwords.words('english'))
wordcloud = WordCloud(background_color="white", stopwords=stop_words).generate(df_train.iloc[0]['review'])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
# plt.show()

print(type(vocabulary.items()))
sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)[:20]

rare_tokens = [(token, count) for token, count in vocabulary.items() if count < 2]
print(rare_tokens)
print(len(rare_tokens))

with open('subjclueslen1-HLTEMNLP05.tff') as f:
    lexicon = f.readlines()

for line in lexicon[:5]:
    print(line)

negative_words = set()
positive_words = set()
for line in lexicon:
    word = re.search(r'word1=(\w+)', line).group(1)
    polarity = re.search(r'priorpolarity=(\w+)', line).group(1)
    if polarity == 'negative':
        negative_words.add(word)
    if polarity == 'positive':
        positive_words.add(word)

print(f'Número de palabras positivas: {len(positive_words)}')
print(f'Número de palabras negativas: {len(negative_words)}')
print(f'10 palabras positivas: {list(positive_words)[:10]}')
print(f'10 palabras negativas: {list(negative_words)[:10]}')

class LexiconVectorizer:
    def __init__(self, positive_words: set, negative_words: set):
        self.positive_words = positive_words
        self.negative_words = negative_words

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        docs = []
        for doc in X:
            docs.append(self.count_words(doc))
        return np.array(docs)

    def count_words(self, text: str) -> "list[int]":
        proccessed_text = tokenize_text(text)
        positive_count = sum([1 for word in proccessed_text if word in self.positive_words])
        negative_count = sum([1 for word in proccessed_text if word in self.negative_words])
        return [positive_count, negative_count]

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def __repr__(self) -> str:
        return "LexiconVectorizer()"

lexicon_vectorizer = LexiconVectorizer(positive_words, negative_words)
X_train_lexicon = lexicon_vectorizer.fit_transform(df_train["review"])
print('Representación vectorial de las primeras 5 reseñas:')
print(X_train_lexicon[:5])

bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(df_train["review"])
print('Representación vectorial de la primera reseña:')
print(X_train_bow[0])

class LexiconCountVectorizer(LexiconVectorizer):
    def __init__(self, positive_words: set, negative_words: set, *args, **kwargs):
        super().__init__(positive_words, negative_words)
        self.count_vect = CountVectorizer(*args, **kwargs)

    def fit(self, X, y=None):
        self.count_vect.fit(X)
        return self

    def transform(self, X):
        bow = self.count_vect.transform(X)
        lexicon_features = []
        for doc in X:
            lexicon_features.append(self.count_words(doc))
        lexicon_features = np.array(lexicon_features) / (np.sum(lexicon_features, axis=1).reshape(-1, 1) + 0.01)
        lexicon_features = scipy.sparse.csr_matrix(lexicon_features)
        return scipy.sparse.hstack([lexicon_features, bow], format='csr')

    def set_params(self, **params):
        self.count_vect.set_params(**params)
        return self

    def __repr__(self) -> str:
        return "LexiconCountVectorizer()"

bow_vectorizer = LexiconCountVectorizer(positive_words, negative_words)
X_train_bow = bow_vectorizer.fit_transform(df_train["review"])
print('Representación vectorial de la primera reseña:')
print(X_train_bow[0])

vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_bow = vectorizer.fit_transform(df_train['review'])
X = X_train_bow.toarray()
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], 
                      X_pca[:, 1], 
                      c=df_train['sentiment'], 
                      cmap='bwr',
                      alpha=0.6)
plt.colorbar(scatter, ticks=[0, 1], label='Sentiment')
plt.title("Visualización con PCA")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
# plt.show()

list_train_reviews = list(df_train["review"])
list_train_labels = list(df_train["sentiment"])

# mlflow.set_tracking_uri('http://localhost:5000')
experiment = mlflow.set_experiment("sklearn-text-classification")

list_train_reviews = list(df_train["review"])
list_train_labels = list(df_train["sentiment"])

pipeline = Pipeline([
    ('representation', CountVectorizer()),
    ('classifier', LogisticRegression())
])

param_grid = [
    {
        'representation': [CountVectorizer(), TfidfVectorizer(), LexiconCountVectorizer(positive_words, negative_words)],
        'representation__max_df': [0.8, 0.9, 1.0],
        'representation__min_df': [0.1, 0.5, 1],
        'classifier': [MultinomialNB(), LogisticRegression(max_iter=1000)]
    }
]

with mlflow.start_run(experiment_id=experiment.experiment_id) as parent_run:
    search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    search.fit(list_train_reviews, list_train_labels)
    
    best_params = search.best_params_
    best_score = search.best_score_
    mlflow.log_param("best_params", str(best_params))
    mlflow.log_metric("best_cv_accuracy", best_score)
    mlflow.sklearn.log_model(search.best_estimator_, "best-model")
    
    cv_results = search.cv_results_
    for i, params in enumerate(cv_results["params"]):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("grid_search_index", i)
            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score", cv_results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", cv_results["std_test_score"][i])
    
    results_df = pd.DataFrame(cv_results)
    results_df.to_csv("cv_results.csv", index=False)
    mlflow.log_artifact("cv_results.csv")
    
    print("Mejor configuración:\n{}\n".format(best_params))
    print("Mejor puntaje de validación cruzada: {:.2f}".format(best_score))

mlflow.sklearn.save_model(search.best_estimator_, "model_export")
print(search.best_estimator_.score(df_test["review"], df_test["sentiment"]))

y_pred = search.best_estimator_.predict(df_test["review"])
conf_mtx = confusion_matrix(df_test["sentiment"], y_pred, normalize='true')
disp = ConfusionMatrixDisplay(conf_mtx, display_labels=["neg", "pos"])
disp.plot()
# plt.show()

print(classification_report(df_test["sentiment"], y_pred, target_names=["neg", "pos"]))

pos_example = "I loved this movie, it was amazing!"
neg_example = "I hated this movie, it was terrible!"
print(f'Ejemplo positivo: {search.best_estimator_.predict([pos_example])[0]}')
print(f'Ejemplo negativo: {search.best_estimator_.predict([neg_example])[0]}')
