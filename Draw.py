from anytree import Node, RenderTree


def draw_syntax_tree(doc) -> None:
    """
    Рисует синтаксическое дерево в консоли с ASCII-графикой.
    """
    nodes = {}

    # Создаем узлы для всех токенов (кроме пунктуации)
    for token in doc:
        if token.dep_ == "punct":
            continue
        nodes[token.i] = Node(f"{token.dep_.upper()}: {token.text}")

    # Строим связи между узлами
    for token in doc:
        if token.dep_ == "punct":
            continue
        current_node = nodes[token.i]
        parent_token = token.head

        # Игнорируем пунктуацию в родителях
        while parent_token.dep_ == "punct" and parent_token != parent_token.head:
            parent_token = parent_token.head

        if parent_token != token and parent_token.i in nodes:
            current_node.parent = nodes[parent_token.i]

    # Находим корневой узел
    root = next((nodes[token.i] for token in doc if
                 token.head == token and token.dep_ != "punct"), None)

    if root:
        for pre, _, node in RenderTree(root):
            print(f"{pre}{node.name}")
    else:
        print("Корневой узел не найден")


'''
from ipymarkup import show_dep_markup
import itertools

def draw_syntax_tree(doc) -> None:
    """
    Рисует синтаксическое дерево для spaCy Doc.
    """
    nodes = [token.text for token in doc]
    edges = []

    for token in doc:
        # Пропускаем пунктуацию и корневую ноду с self-loop
        if token.dep_ != 'punct' and token.head != token:
            edges.append(
                (token.head.i,
                 token.i,
                 token.dep_)
            )

    # Визуализация с помощью ipymarkup
    show_dep_markup(nodes, edges)


from spacy import displacy
import tempfile
import webbrowser


def draw_syntax_tree(doc) -> None:
    """
    Рисует синтаксическое дерево для spaCy Doc и открывает его в браузере.
    """
    # Генерируем HTML-код для визуализации
    html = displacy.render(doc, style="dep", page=True, jupyter=False)

    # Сохраняем HTML во временный файл
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w",
                                     encoding="utf-8") as f:
        f.write(html)
        filename = f.name

    # Открываем файл в браузере
    webbrowser.open(f"file://{filename}")

'''