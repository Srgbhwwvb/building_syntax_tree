from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from proccess_latex import replace_latex

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")

semantic_classes = {
    'define': {
        'называть','называется','называются', 'называться',
        'определять', 'определяться', 'есть', 'быть', 'являться'
    },
    'exists': {
        'существовать','существует', 'найтись',
        'иметься', 'выделить', 'можно'
    },
    'foralls': {'все', 'любой', 'любого','всех', 'весь', 'произвольный'},
    'iff': {'если'},
    'than': {'то'}
}
target_phrases = {
    'такой что', 'такой что:', 'так что',
    'для который выполняться', 'выполняться условие:',
    'выполняться условия:', 'выполнять условие:', 'верно'
}

class my_object:
    def __init__(self, Type, Text):
        self.Type = Type
        self.Text = Text
        if Text == '':
            self.Text = self.Type
def pprint(sem):
    for i in sem:
        print(i.Type, ':', i.Text)

def get_word_embedding(word):

    inputs = tokenizer(
        word,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Получение эмбеддингов субтокенов
    subword_embeddings = outputs.last_hidden_state[0]

    # Усреднение по всем субтокенам слова
    return torch.mean(subword_embeddings, dim=0).numpy()



# def get_word_embeddings(text):
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     outputs = model(**inputs)
#
#     words = text.split()
#     word_embeddings = []
#     word_ids = inputs.word_ids(batch_index=0)  # Для первого элемента батча
#
#     current_word_id = None
#     embeddings_buffer = []
#
#     for i, word_id in enumerate(word_ids):
#         if word_id is None:  # Пропускаем служебные токены
#             continue
#
#         if word_id != current_word_id:
#             # Сохраняем предыдущий эмбеддинг (если был)
#             if embeddings_buffer:
#                 word_embeddings.append(np.mean(embeddings_buffer, axis=0))
#                 embeddings_buffer = []
#             current_word_id = word_id
#
#         # Добавляем эмбеддинг субтокена в буфер
#         embeddings_buffer.append(outputs.last_hidden_state[0, i].detach().numpy())
#
#     # Добавляем последнее слово
#     if embeddings_buffer:
#         word_embeddings.append(np.mean(embeddings_buffer, axis=0))
#
#     return word_embeddings


def classify_tokens(emb_operators, sentence_embeddings, words, threshold=0.8):

    # Нормализация эмбеддингов
    def normalize(emb):
        return emb / np.linalg.norm(emb) if np.any(emb) else emb

    # Подготовка операторов
    operators = {
        k: normalize(v)
        for k, v
        in emb_operators.items()
        if v is not None
    }

    result = []
    for word, word_emb in zip(words, sentence_embeddings):
        word_emb = normalize(word_emb)
        max_sim = -1
        best_op = 'str'

        # Вычисление сходства с каждым оператором
        for op_name, op_emb in operators.items():
            if op_emb is None:
                continue

            sim = cosine_similarity([word_emb], [op_emb])[0][0]
            if sim > max_sim:
                max_sim = sim
                best_op = op_name

        # Принятие решения на основе порога
        if max_sim > threshold:
            result.append(my_object(best_op, word))
        else:
            result.append(my_object('str', word))

    return result


def prepare_phrase_embeddings(phrases):
    phrase_embeddings = {}

    for phrase in phrases:
        clean_phrase = phrase.replace(':', '').strip()
        words = clean_phrase.split()

        # Статический эмбеддинг (усреднение слов)
        static_emb = np.mean(
            [get_word_embedding(word) for word in words],
            axis=0
        )

        # Контекстный эмбеддинг (обработка как последовательности)
        inputs = tokenizer(
            clean_phrase,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
        contextual_emb = outputs.last_hidden_state[0].mean(dim=0).numpy()

        phrase_embeddings[phrase] = {
            'static': static_emb,
            'contextual': contextual_emb,
            'length': len(words)
        }

    return phrase_embeddings


# 2. Функция поиска фраз в предложении
def find_phrases(text, phrase_embeddings, threshold=0.75):
    words = text.split()
    n = len(words)
    matches = []

    # Генерация всех возможных окон
    for i in range(n):
        for j in range(i + 1, min(i + 5, n + 1)):
            window = words[i:j]
            window_len = j - i

            # Пропускаем слишком короткие/длинные окна
            valid_phrases = [
                p for p, data in phrase_embeddings.items()
                if data['length'] == window_len
            ]
            if not valid_phrases:
                continue

            # Вычисляем эмбеддинги для окна
            # Статический
            static_emb = np.mean(
                [get_word_embedding(w) for w in window],
                axis=0
            )

            # Контекстный
            inputs = tokenizer(
                ' '.join(window),
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            with torch.no_grad():
                outputs = model(**inputs)
            contextual_emb = outputs.last_hidden_state[0].mean(dim=0).numpy()

            # Вычисляем комбинированное сходство
            max_sim = -1
            best_phrase = None
            for phrase in valid_phrases:
                # Взвешенное среднее статического и контекстного сходства
                static_sim = cosine_similarity(
                    [static_emb],
                    [phrase_embeddings[phrase]['static']]
                )[0][0]

                context_sim = cosine_similarity(
                    [contextual_emb],
                    [phrase_embeddings[phrase]['contextual']]
                )[0][0]

                combined_sim = 0.4 * static_sim + 0.6 * context_sim

                if combined_sim > max_sim:
                    max_sim = combined_sim
                    best_phrase = phrase

            if max_sim > threshold:
                matches.append({
                    'phrase': best_phrase,
                    'start': i,
                    'end': j,
                    'score': max_sim
                })

    # Фильтрация пересекающихся совпадений
    matches.sort(key=lambda x: x['score'], reverse=True)
    final_matches = []
    used_positions = set()

    for match in matches:
        positions = set(range(match['start'], match['end']))
        if not positions & used_positions:
            final_matches.append(match)
            used_positions.update(positions)

    return final_matches

# Создание словаря для хранения эмбеддингов
class_embeddings = {}

for class_name, words in semantic_classes.items():
    embeddings = []

    for word in words:
        try:
            # Получение эмбеддинга слова
            word_emb = get_word_embedding(word)
            embeddings.append(word_emb)
        except Exception as e:
            print(f"Ошибка при обработке слова '{word}': {str(e)}")
            continue

    if embeddings:
        # Усреднение по всем словам класса
        class_embeddings[class_name] = np.mean(embeddings, axis=0)
    else:
        class_embeddings[class_name] = None
        print(f"Не удалось получить эмбеддинги для класса {class_name}")

phrase_embeddings = prepare_phrase_embeddings(target_phrases)

text, _ = replace_latex("Функция $f(x)$ называется непрерывной в точке $a$, если для любого $\\varepsilon > 0$ существует $\\delta > 0$ такое, что для всех $x$ выполняется условие: если $|x - a| < \\delta$, то $|f(x) - f(a)| < \\varepsilon$.")
sentence_embeddings=[]
arr = text.split(" ")
for x in arr:
    sentence_embeddings.append(get_word_embedding(x))
#sentence_embeddings = get_word_embeddings(text)
#print(f"Количество слов: {len(text.split())}, эмбеддингов: {len(sentence_embeddings)}")

for class_name, emb in class_embeddings.items():
    if emb is None:
        print(f"{class_name}: embedding not available")


classified_objects = classify_tokens(
    class_embeddings,
    sentence_embeddings,
    words=text.split()
)
pprint(classified_objects)

results = find_phrases(text, phrase_embeddings)

# Вывод результатов
print("Найденные фразы:")
for res in results:
    phrase = ' '.join(text.split()[res['start']:res['end']])
    print(f"Фраза: {phrase} | Совпадение: {res['phrase']} | Сходство: {res['score']:.2f}")
