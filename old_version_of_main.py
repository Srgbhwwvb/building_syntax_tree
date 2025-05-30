from proccess_latex import replace_latex
from draw_tree import draw_syntax_tree
import spacy
from spacy import displacy
from nltk.tokenize import sent_tokenize
import nltk
#nltk.download('punkt_tab')

examples=["Функция $f(x)$ называется непрерывной в точке $a$, если для любого $\\varepsilon > 0$ существует $\\delta > 0$ такое, что для всех $x$ выполняется условие: если $|x - a| < \\delta$, то $|f(x) - f(a)| < \\varepsilon$.",
          "Пусть $V$ и $W$ — векторные пространства над полем $F$. Отображение $T: V \\to W$ называется линейным оператором, если для любых векторов $u, v \\in V$ и скаляра $alpha \\in F$ выполняются условия: $T(u + v) = T(u) + T(v)$, $T(\\alpha u) = \\alpha T(u)$. ",
          "Подмножество $K$ метрического пространства $(X, d)$ называется компактным, если из любого открытого покрытия $K$ можно выбрать конечное подпокрытие.",
          "Топологическое пространство $M$ называется дифференцируемым многообразием размерности $n$, если оно локально гомеоморфно $\\mathbb{R}^n$ и переходные функции между картами гладкие (бесконечно дифференцируемые).",
          "Гауссовская кривая — это кривая, задаваемая уравнением: $y = \\frac{1}{\\sigma \\sqrt{2\\pi}} e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}$ где $\\mu$ — среднее значение, а $\\sigma$ — стандартное отклонение.",
          "Если функция $f(x)$ непрерывна на интервале $[a, b]$, дифференцируема на интервале $(a, b)$ и $f(a) = f(b)$, то существует точка $c \\in (a, b)$ такая, что $f'(c) = 0$.",
          "Если функция $f(x)$ непрерывна на интервале $[a, b]$ и дифференцируема на интервале $(a, b)$, то существует точка $c \\in (a, b)$ такая, что: $f'(c) = \\frac{f(b) - f(a)}{b - a}$.",
          "Если последовательность $\\{x_n\\}$ монотонна и ограничена, то она сходится к некоторому пределу.",
          "Пусть $f(x, t)$ — функция, определенная на прямоугольнике $[a, b] \\times [c, d]$, непрерывная по $t$ и имеющая непрерывную производную по $x$. Тогда интеграл: $F(x) = \\int_{c}^{d} f(x, t) dt $ дифференцируем на $[a, b]$, и его производная равна: $F'(x) = \\int_{c}^{d} \\frac{\\partial f}{\\partial x}(x, t) dt$.",
          "Пусть $A$ — самосопряженный линейный оператор в гильбертовом пространстве $H$. Тогда существует унитарный оператор $U$ и диагонализация $D$ такая, что: $ A = UDU^{-1}$, где $D$ — диагональная матрица с собственными значениями $A$ на диагонали.",
          "Функция $\\mu: \\Sigma \\to[0, +\\infty]$ называется мерой на измеримом пространстве $(X, \\Sigma)$, если $\\mu(\\varnothing) = 0$ и для любых дизъюнктных $A_n \\in \\Sigma$ выполняется $\\mu\\left(\\bigcup_{ n = 1 } ^ \\infty A_n\\right) = \\sum_{ n = 1 } ^ \\infty \\mu(A_n)$.",
          "Семейство $\\Sigma \\subseteq 2 ^ X$ является $\\sigma$ - алгеброй, если содержит $X$, замкнуто относительно дополнений и счётных объединений.",
          "Функция $f : X \\to \\mathbb{ R }$ измерима относительно $\\Sigma$, когда ${ x \\in X \\mid f(x) < a } \\in \\Sigma$ для любого $a \\in \\mathbb{ R }$.",
          "Функция $\\mu^ : 2 ^ X \\to[0, +\\infty]$ является внешней мерой, если $\\mu ^ (\\varnothing) = 0$, $\\mu ^ (A) \\leq \\mu ^ (B)$ при $A \\subseteq B$, и $\\mu^ \\left(\\bigcup_{ n = 1 }^ \\infty A_n\\right) \\leq \\sum_{ n = 1 }^ \\infty \\mu ^ (A_n)$.",
          "Тройка $(X, \\Sigma, \\mu)$ образует пространство с мерой, где $\\Sigma$ — $\\sigma$ - алгебра, а $\\mu$ — счётно - аддитивная мера.",
          "Мера $\\mu$ полна, если каждое подмножество множества нулевой меры принадлежит $\\Sigma$ и имеет нулевую меру.",
          "Мерой Лебега на $\\mathbb{ R }^ n$ называется единственная счётно - аддитивная мера, инвариантная относительно сдвигов и нормированная условием $\\lambda(^ n) = 1$.",
          "Борелевской $\\sigma$ - алгеброй на топологическом пространстве $X$ называется минимальная $\\sigma$ - алгебра, содержащая все открытые множества.",
          "Последовательность функций $f_n$ сходится почти всюду к $f$, если $\\mu\\left({ x \\mid \\lim_{n \\to \\infty} f_n(x) \\neq f(x) }\\right) = 0$.",
          "Интеграл Лебега от неотрицательной измеримой $f$ определяется как $\\int_X f, d\\mu = \\sup \\left{ \\sum_{i = 1} ^ n y_i \\mu(A_i) \\mid \\sum_{i = 1} ^ n y_i \\chi_{A_i} \\leq f \\right }$.",
          "Если $0 \\leq f_n(x) \\uparrow f(x)$ почти всюду, то $\\lim_{ n \\to \\infty } \\int_X f_n, d\\mu = \\int_X f, d\\mu$.",
          "Для последовательности неотрицательных измеримых функций $\\liminf_{ n \\to \\infty } \\int_X f_n, d\\mu \\geq \\int_X \\liminf_{ n \\to \\infty } f_n, d\\mu$.",
          "Если $ | f_n | \\leq g$ для интегрируемой $g$ и $f_n \\to f$ почти всюду, то $\\lim_{ n \\to \\infty } \\int_X | f_n - f | , d\\mu = 0$.",
          "Если $\\sigma$ - конечная мера $\\nu$ абсолютно непрерывна относительно $\\mu$, то существует функция $f \\geq 0$ такая, что $\\nu(A) = \\int_A f, d\\mu$.",
          "Для любой измеримой функции $f : [a, b] \\to \\mathbb{ R }$ и $\\varepsilon > 0$ найдётся непрерывная $g$, совпадающая с $f$ вне множества меры меньше $\\varepsilon$.",
          "Семейство $\\Sigma = { A \\subseteq X \\mid \\mu ^ (E) = \\mu ^ (E \\cap A) + \\mu ^ (E \\setminus A) , \\forall E \\subseteq X }$ является $\\sigma$ - алгеброй, на которой внешняя мера $\\mu^ $ счётно - аддитивна.",
          "Если функция $f$ интегрируема на $X \\times Y$, то $\\int_{ X \\times Y } f, d(\\mu \\otimes \\nu) = \\int_X \\left(\\int_Y f(x, y), d\\nu \\right) d\\mu$.",
          "Любой положительный линейный функционал на пространстве $C_c(X)$ представим в виде $\\Lambda(f) = \\int_X f, d\\mu$ для некоторой меры $\\mu$.",
          "Если $f_n \\to f$ почти всюду на множестве конечной меры, то для любого $\\varepsilon > 0$ существует подмножество $A$ с $\\mu(A) < \\varepsilon$, на котором сходимость равномерна.",
          "$\\sigma$ - Конечная мера, заданная на полукольце множеств, единственным образом продолжается до меры на порождённой $\\sigma$ - алгебре.",
          "Топологическое пространство связно тогда и только тогда, когда оно не может быть представлено в виде объединения двух непустых непересекающихся открытых множеств."]

define = {'называть', 'называться', 'определять', 'определяться', 'есть', 'быть', 'являться'}
exists = {'существовать', 'найтись', 'иметься', 'выделить', 'можно'}
foralls = {'все', 'любой', 'всех', 'произвольный'}
impls = {'выполняться', 'верно'}


class Node:
    def __init__(self, name, i=0, suc=None, syntax=None, head=0):
        self.name = name
        self.syntax = syntax
        self.suc = suc if suc is not None else []
        self.head = head

    def __repr__(self):
        return f"{self.name} {self.suc}" if self.suc else f"{self.name}"


def get_children_by_dep(token, dep):
    return [child for child in token.children if child.dep_ in dep]


def get_name(start_token, semantic):
    operators = {'forall', 'forall_', 'exist', 'exist_', 'eqv', 'eqv_', 'impl', 'impl_'}
    tokens = []
    stack = [start_token]
    while stack:
        token = stack.pop()
        if semantic[token.i] in operators:
            break
        if token.text in "'.,;:?!если":
            break
        if token not in tokens:
            tokens.append(token)
            stack.extend(reversed(list(token.children)))

    sorted_tokens = sorted(tokens, key=lambda x: x.i)
    return ' '.join(t.lemma_ for t in sorted_tokens)


def proccess_eqv(root, tree, semantic):
    '''
    Если есть root -> xcomp: создается поддерево (root -> nsubj) is (root -> xcomp). is -- левый потомок eqv.
    Иначе: (root -> nsubj) eqv ()
    '''
    nsubj_ = get_children_by_dep(root, 'nsubj')[0]
    xcomp_ = get_children_by_dep(root, 'xcomp')[0]

    if xcomp_:
        nsubj = Node(get_name(nsubj_, semantic), syntax=nsubj_)
        xcomp = Node(get_name(xcomp_, semantic), syntax=xcomp_)
        is_ = Node('is', suc=[nsubj, xcomp])
        eqv = Node('eqv', suc=[is_], syntax=root)
        tree.extend([eqv, is_, nsubj, xcomp])
        # print("after    ", tree)
    else:
        nsubj = Node(get_name(nsubj_, semantic), syntax=nsubj_)
        eqv = Node('eqv', suc=[nsubj], syntax=root)
        tree.extend([eqv, nsubj])

    semantic[root.i] = 'eqv'
    return tree, eqv


def proccess_forall(root, head, tree, semantic):
    '''
    Примеряет шаблон: "det.lemma in foralls". Создает узел forall и узел субъекта forall. Возвращает дерево и ссылку на forall
    forall (det.head) ()
    '''
    nsubj_ = root.head
    nsubj = Node(get_name(nsubj_, semantic), syntax=nsubj_)
    forall = Node('forall', suc=[nsubj], syntax=root)
    tree.extend([forall, nsubj])
    head.suc.append(forall)
    semantic[root.i] = 'forall'
    return tree, forall


def proccess_exist(root, head, tree, semantic):
    '''
    Субъект: exist -> nsubj -> nummod
    Проверяется наличие впереди стоящего квантора всеобщности с помощью шаблона: exist -> nsubj -> det & det.lemma in foralls.
    Проверяется наличие parataxis.
    Возвращает дерево, ссылку на exist и (если есть, parataxis)
    forall (exist -> nsubj -> det.head) (exist (exist -> nsubj -> nummod) ( ) )
    '''
    nsubj_ = get_children_by_dep(root, {'nsubj', 'obj', 'obl'})
    nsubj_nummod = None
    nsubj_det = None
    idx = 1
    for i in range(len(nsubj_)):
        # print(get_children_by_dep(nsubj_[i], 'nummod'))
        if (get_children_by_dep(nsubj_[i], 'nummod')):
            nsubj_nummod = get_children_by_dep(nsubj_[i], 'nummod')[0]
        if (get_children_by_dep(nsubj_[i], 'det')):
            nsubj_det = get_children_by_dep(nsubj_[i], 'det')[0]
            idx = i

    if (not nsubj_nummod):
        print(nsubj_)
        print((idx + 1) % 2)
        if (len(nsubj_) < 2):
            nsubj_nummod = nsubj_[0]
        else:
            nsubj_nummod = nsubj_[(idx + 1) % 2]

    nsubj_ex = Node(get_name(nsubj_nummod, semantic), syntax=nsubj_nummod)

    # если перед существованием стоит квантор всеобщности
    if (nsubj_det):
        if (semantic[nsubj_det.i] in 'forall_'):
            nsubj0_ = nsubj_det.head
            nsubj = Node(get_name(nsubj0_, semantic), syntax=nsubj0_)
            exist = Node('exist', suc=[nsubj_ex])
            forall = Node('forall', suc=[nsubj, exist], syntax=nsubj_det)
            tree.extend([forall, nsubj_ex, nsubj, exist])
            head.suc.append(forall)
            semantic[nsubj_det.i] = 'forall'
    else:
        exist = Node('exist', suc=[nsubj_ex])
        tree.extend([nsubj_ex, exist])
        head.suc.append(exist)
    # for parataxis
    parataxis_ = get_children_by_dep(root, 'parataxis')
    parataxis_ = parataxis_[0] if len(parataxis_) > 0 else None
    semantic[root.i] = 'exist'

    return tree, exist, parataxis_


def proccess_impl(root, head, tree, parataxis, semantic):
    # костыль под первый пример. Лучше через выделение "если ... то ..."
    '''
    (parataxis -> advcl) impl (parataxis -> nsubj)
    '''
    obl_ = get_children_by_dep(root, 'obl')[0]
    det_ = get_children_by_dep(obl_, 'det')[0]

    if parataxis:
        advcl_ = get_children_by_dep(parataxis, 'advcl')[0]
        if advcl_:
            nsubj = Node(get_name(parataxis, semantic), syntax=parataxis)
            advcl = Node(get_name(advcl_, semantic), syntax=advcl_)
            impl = Node('impl', suc=[advcl, nsubj])
            tree.extend([impl, nsubj, advcl])
        else:
            nsubj = Node(get_name(root, semantic), syntax=root)
            impl = Node('impl', suc=[nsubj])
            tree.extend([impl, nsubj])
        parataxis = None

        if (semantic[det_.i] == 'forall_'):
            obl = Node(get_name(obl_, semantic), syntax=obl_)
            forall = Node('forall', suc=[obl, impl], syntax=det_)
            tree.extend([forall, obl])
            head.suc.append(forall)
            semantic[det_.i] = 'forall'
        else:
            head.suc.append(impl)

        semantic[root.i] = 'impl'
        return tree, impl
    else:
        return tree, None

def if_so(text):
    if_start = text.lower().find('если')
    if(if_start==0):
        if_end =  text.find('то')
        if if_start < if_end:
            between_if_then = text[if_start + 5:if_end-2]
            after_then = text[if_end + 3:]  # +2 чтобы не включать "то"

            return between_if_then, after_then
    return 0




# Семантическая маска предложения
def give_semantic(doc, sent):
    semantic = ['0'] * len(doc)
    for i, word in enumerate(doc):
        if word.lemma_ in define:
            semantic[i] = 'eqv_'
        elif word.lemma_ in foralls:
            semantic[i] = 'forall_'
        elif word.lemma_ in exists:
            semantic[i] = 'exist_'
        elif '+' in word.text:
            semantic[i] = 'formula_'
        elif word.lemma_ in impls and 'если' in sent.lower():
            semantic[i] = 'impl_'
        else:
            pass
    return semantic


# Функция, возвращающая ближайший непросмотренный оператор (если такой имеется)
from collections import deque
def next_operator(root, semantic):
    if root is None:
        return 0
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if '_' in semantic[node.i]:  # если слово -- оператор/формула, которую еще не вставили в ДВ
            return node
        for child in node.children:
            queue.append(child)
    return 0


def template(root, head, semantic, tree, parataxis) -> None:
    """

    """
    # Как выбирать head для вершины?
    new_root = head
    if semantic[root.i] == 'eqv_':
        tree, new_root = proccess_eqv(root, tree, semantic)

    elif semantic[root.i] == 'forall_':
        tree, new_root = proccess_forall(root, head, tree, semantic)

    elif semantic[root.i] == 'exist_':
        tree, new_root, parataxis = proccess_exist(root, head, tree, semantic)

    elif semantic[root.i] == 'impl_':
        tree, _ = proccess_impl(root, head, tree, parataxis, semantic)

    elif root.dep_ == 'conj':
        # обработка однородных членов (добавление 'and')
        pass

    elif semantic[root.i] == 'if_':
        # обработка однородных членов (добавление 'and')
        pass

    elif semantic[root.i] == 'and':
        # обработка однородных членов (добавление 'and')
        pass

    # вариант по умолчанию
    else:
        node = Node(str(get_name(root, semantic)))
        tree.append(node)

        # suc = [x for x in root.children]
        # Name = semantic[root.i] if semantic[root.i] != '0' else root.text
        # Node(name = Name, suc = suc)

    return tree, new_root, parataxis



# Рекурсивная функция построения ДВ начиная с корня
def build(root, head, semantic, tree, parataxis):
    # if ("если ... то.." разделяет предложение на части без остатка) -> теорема-импликац
    subtree = tree
    #if next_operator(root, semantic):
    subtree, new_head, new_parataxis = template(root, head, semantic, tree, parataxis)
    # print(tree[0])
    for suc in root.children:
        build(suc, new_head, semantic, tree, new_parataxis)
    return subtree


# Рекурсивная функция построения ДВ начиная с листьев. Типа DFS
def build2(root, semantic, tree):
    for word in root.children:
        if not next_operator(word, semantic):
            subtree = template(word, semantic)
        else:
            subtree = build(word, semantic)
    return subtree


# MAIN
def proccess(text):
    nlp = spacy.load("ru_core_news_lg")
    for sent in text:
        print(sent)
        sent, _ = replace_latex(sent)
        sents = sent_tokenize(sent, language='russian')
        sent = sents[-1]
        parts_if = if_so(sent)
        if(parts_if): #сли теорема вида "Если ... то ..."
            #print(parts)
            sent1, sent2 = parts_if
            doc1, doc2 = nlp(sent1), nlp(sent2)
            root1, root2 = [t for t in doc1 if t.dep_ == 'ROOT'][0], [t for t in doc2 if t.dep_ == 'ROOT'][0]
            semantic1, semantic2  = give_semantic(doc1, sent1), give_semantic(doc2, sent2)
            tree1, tree2 = build(root1, None, semantic1, [], None), build(root2, None, semantic2, [], None)
            impl = Node('impl', suc=[tree1[0], tree2[0]])
            tree = []
            tree.extend([impl, tree1[0], tree2[0]])
            print(" Finally: ")
            print(tree[0])
            return

        else:
            doc = nlp(sent)
            #draw_syntax_tree(doc)
            root = [t for t in doc if t.dep_ == 'ROOT'][0]
            semantic = give_semantic(doc, sent)
            tree = build(root, None, semantic, [], None)
            print(" Finally: ")
            print(tree[0])
        # рисунок


proccess([examples[1]])
#print(if_so("Если функция $f(x)$ непрерывна на интервале $[a, b]$ и дифференцируема на интервале $(a, b)$, то существует точка $c \\in (a, b)$ такая, что: $(f'(c) = \\frac{f(b) - f(a)}{b - a}$"))
# print(nlp('функция +1 непрерывна на интервале +2 и дифференцируема на интервале +3.'))
# print(nlp('существует точка +4 такая, что: +5.'))
