

from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity
from pymorphy2 import MorphAnalyzer  # Для лемматизации русских слов

model = FastText.load_fasttext_format('C:\\Users\\Alex\\Downloads\\185\\model.bin')

class_words = {
    1: {'все', 'любой', 'всех', 'весь', 'произвольный'},
    2: {'существует', 'найдется', 'имеется'},
    3: {'эквивалентно', 'равносильно', 'тогда_и_только_тогда'},
    4: {'влечет', 'следовательно', 'поэтому', 'значит'},
    5: {'и', 'а_также'},
    6: {'или', 'либо'},
    7: {}  # Остальные слова
}

# 3. Создание эталонных эмбеддингов для классов
class_embeddings = {}
for class_id, words in class_words.items():
    embeddings = []
    for word in words:
        try:
            # Лемматизация для улучшения сопоставления
            morph = MorphAnalyzer()
            lemma = morph.parse(word)[0].normal_form
            embeddings.append(model.wv[lemma])
        except KeyError:
            continue
    if embeddings:
        class_embeddings[class_id] = np.mean(embeddings, axis=0)
    else:
        class_embeddings[class_id] = None


# 4. Функция для обработки предложения
def get_semantic_mask(sentence):
    morph = MorphAnalyzer()
    tokens = sentence.lower().split()  # Простая токенизация
    mask = []

    for token in tokens:
        # Лемматизация текущего слова
        lemma = morph.parse(token)[0].normal_form

        try:
            word_vector = model.wv[lemma]
        except KeyError:
            mask.append(7)  # Неизвестное слово
            continue

        max_sim = -np.inf
        best_class = 7  # Класс по умолчанию

        for class_id, class_vec in class_embeddings.items():
            if class_vec is None or class_id == 7:
                continue

            # Вычисление косинусного сходства
            sim = cosine_similarity([word_vector], [class_vec])[0][0]

            if sim > max_sim:
                max_sim = sim
                best_class = class_id

        mask.append(best_class)

    return mask


# 5. Пример использования
sentence = "Любой объект существует и влечет эквивалентность"
mask = get_semantic_mask(sentence)
print(f"Предложение: {sentence}")
print(f"Семантическая маска: {mask}")
