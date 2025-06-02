from proccess_latex import replace_latex
from draw_tree import draw_syntax_tree
import spacy
from spacy import displacy
from nltk.tokenize import sent_tokenize
import nltk
from print_expression_tree import print_tree
#nltk.download('punkt_tab')

examples=[r"Функция $f(x)$ называется непрерывной в точке $a$, если для любого $\varepsilon > 0$ существует $\delta > 0$ такое, что для всех $x$ выполняется условие: если $|x - a| < \delta$, то $|f(x) - f(a)| < \varepsilon$.",
          r"Пусть $V$ и $W$ — векторные пространства над полем $F$. Отображение $T: V \to W$ называется линейным оператором, если для любых векторов $u, v \in V$ и скаляра $alpha \in F$ выполняются условия: $T(u + v) = T(u) + T(v)$, $T(\alpha u) = \alpha T(u)$. ",
          r"Подмножество $K$ метрического пространства $(X, d)$ называется компактным, если из любого открытого покрытия $K$ можно выбрать конечное подпокрытие.",
          r"Топологическое пространство $M$ называется дифференцируемым многообразием размерности $n$, если оно локально гомеоморфно $\mathbb{R}^n$ и переходные функции между картами гладкие (бесконечно дифференцируемые).",
          r"Гауссовская кривая — это кривая, задаваемая уравнением: $y = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ где $\mu$ — среднее значение, а $\sigma$ — стандартное отклонение.",
          r"Если функция $f(x)$ непрерывна на интервале $[a, b]$, дифференцируема на интервале $(a, b)$ и $f(a) = f(b)$, то существует точка $c \in (a, b)$ такая, что $f'(c) = 0$.",
          r"Если функция $f(x)$ непрерывна на интервале $[a, b]$ и дифференцируема на интервале $(a, b)$, то существует точка $c \in (a, b)$ такая, что: $f'(c) = \frac{f(b) - f(a)}{b - a}$.",
          r"Если последовательность $\{x_n\}$ монотонна и ограничена, то она сходится к некоторому пределу.",
          r"Пусть $f(x, t)$ — функция, определенная на прямоугольнике $[a, b] \times [c, d]$, непрерывная по $t$ и имеющая непрерывную производную по $x$. Тогда интеграл: $F(x) = \int_{c}^{d} f(x, t) dt $ дифференцируем на $[a, b]$, и его производная равна: $F'(x) = \int_{c}^{d} \frac{\partial f}{\partial x}(x, t) dt$.",
          r"Пусть $A$ — самосопряженный линейный оператор в гильбертовом пространстве $H$. Тогда существует унитарный оператор $U$ и диагонализация $D$ такая, что: $ A = UDU^{-1}$, где $D$ — диагональная матрица с собственными значениями $A$ на диагонали.",
          r"Функция $\mu: \Sigma \to[0, +\infty]$ называется мерой на измеримом пространстве $(X, \Sigma)$, если $\mu(\varnothing) = 0$ и для любых дизъюнктных $A_n \in \Sigma$ выполняется $\mu\left(\bigcup_{ n = 1 } ^ \infty A_n\right) = \sum_{ n = 1 } ^ \infty \mu(A_n)$.",
          r"Семейство $\Sigma \subseteq 2 ^ X$ является $\sigma$ - алгеброй, если содержит $X$, замкнуто относительно дополнений и счётных объединений.",
          r"Функция $f : X \to \mathbb{ R }$ измерима относительно $\Sigma$, когда ${ x \in X \mid f(x) < a } \in \Sigma$ для любого $a \in \mathbb{ R }$.",
          r"Функция $\mu^ : 2 ^ X \to[0, +\infty]$ является внешней мерой, если $\mu ^ (\varnothing) = 0$, $\mu ^ (A) \leq \mu ^ (B)$ при $A \subseteq B$, и $\mu^ \left(\bigcup_{ n = 1 }^ \infty A_n\right) \leq \sum_{ n = 1 }^ \infty \mu ^ (A_n)$.",
          r"Тройка $(X, \Sigma, \mu)$ образует пространство с мерой, где $\Sigma$ — $\sigma$ - алгебра, а $\mu$ — счётно - аддитивная мера.",
          r"Мера $\mu$ полна, если каждое подмножество множества нулевой меры принадлежит $\Sigma$ и имеет нулевую меру.",
          r"Мерой Лебега на $\mathbb{ R }^ n$ называется единственная счётно - аддитивная мера, инвариантная относительно сдвигов и нормированная условием $\lambda(^ n) = 1$.",
          r"Борелевской $\sigma$ - алгеброй на топологическом пространстве $X$ называется минимальная $\sigma$ - алгебра, содержащая все открытые множества.",
          r"Последовательность функций $f_n$ сходится почти всюду к $f$, если $\mu\left({ x \mid \lim_{n \to \infty} f_n(x) \neq f(x) }\right) = 0$.",
          r"Интеграл Лебега от неотрицательной измеримой $f$ определяется как $\int_X f, d\mu = \sup \left{ \sum_{i = 1} ^ n y_i \mu(A_i) \mid \sum_{i = 1} ^ n y_i \chi_{A_i} \leq f \right }$.",
          r"Если $0 \leq f_n(x) \uparrow f(x)$ почти всюду, то $\lim_{ n \to \infty } \int_X f_n, d\mu = \int_X f, d\mu$.",
          r"Для последовательности неотрицательных измеримых функций $\liminf_{ n \to \infty } \int_X f_n, d\mu \geq \int_X \liminf_{ n \to \infty } f_n, d\mu$.",
          r"Если $ | f_n | \leq g$ для интегрируемой $g$ и $f_n \to f$ почти всюду, то $\lim_{ n \to \infty } \int_X | f_n - f | , d\mu = 0$.",
          r"Если $\sigma$ - конечная мера $\nu$ абсолютно непрерывна относительно $\mu$, то существует функция $f \geq 0$ такая, что $\nu(A) = \int_A f, d\mu$.",
          r"Для любой измеримой функции $f : [a, b] \to \mathbb{ R }$ и $\varepsilon > 0$ найдётся непрерывная $g$, совпадающая с $f$ вне множества меры меньше $\varepsilon$.",
          r"Семейство $\Sigma = { A \subseteq X \mid \mu ^ (E) = \mu ^ (E \cap A) + \mu ^ (E \setminus A) , \forall E \subseteq X }$ является $\sigma$ - алгеброй, на которой внешняя мера $\mu^ $ счётно - аддитивна.",
          r"Если функция $f$ интегрируема на $X \times Y$, то $\int_{ X \times Y } f, d(\mu \otimes \nu) = \int_X \left(\int_Y f(x, y), d\nu \right) d\mu$.",
          r"Любой положительный линейный функционал на пространстве $C_c(X)$ представим в виде $\Lambda(f) = \int_X f, d\mu$ для некоторой меры $\mu$.",
          r"Если $f_n \to f$ почти всюду на множестве конечной меры, то для любого $\varepsilon > 0$ существует подмножество $A$ с $\mu(A) < \varepsilon$, на котором сходимость равномерна.",
          r"$\sigma$ — конечная мера, заданная на полукольце множеств, единственным образом продолжается до меры на порождённой $\sigma$ - алгебре."]

define = {'называть', 'называться', 'определять', 'определяться', 'есть', 'быть', 'являться', 'называется'}
exists = {'существовать', 'найтись', 'иметься', 'выделить', 'можно'}
foralls = {'все', 'любой', 'всех', 'весь', 'произвольный'}
iff = {'если', 'когда'}
and_ = {'и'}
such = {
    'такой что', 'такой что:', 'так что', 'для который выполняться', 'выполняться условие:',
    'выполняться условия:', 'выполнять условие:', 'верно'
}

class Node:
    def __init__(self, name, i=0, suc=None, syntax=None, head=0):
        self.name = name
        self.suc = suc if suc is not None else []

    def __repr__(self):
        return f"{self.name} {self.suc}" #if self.suc else f"{self.name}"


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


def if_then(text, _):
    """
    Выделяет шаблон "__ если __ (то) __ "
    """
    if_start = text.lower().find('если')
    if if_start == -1:
        if_start = text.lower().find('когда')
    if if_start > -1:
        if_end = text.find(' то ')
        if if_start < if_end:
            between_if_then = text[(if_start + 5):(if_end-1)]
            after_then = text[(if_end + 4):]  # +2 чтобы не включать "то"
            return between_if_then, after_then
        else:
            return text[: if_start-1], text[if_start+6 :]
    return '0', '0'


def define_if(nlp, doc, text, semantic):
    """
    Выделяет шаблон вида "__ называется __ (если) __"
    """
    root = [t for t in doc if t.dep_ == 'ROOT'][0]
    if semantic[root.i] == 'eqv_':
        define_start = root.i
        define_end = text.find('если')
        if(define_end > -1):
            if define_start < define_end:
                between_def_if = text[0:define_end-1]
                after_if = text[define_end + 5:]  # +2 чтобы не включать "то"
                return between_def_if, after_if
        else:
            text_arr = text.split(' ')
            for i in range(len(text_arr)):
                print(doc[i].lemma_, text_arr[i])
                if doc[i].lemma_ in define:
                    before_define = text_arr[:i]
                    after_define = text_arr[i+1:]
                    return (' ').join(before_define), (' ').join(after_define)
                break
    return '0', '0'


def dash(doc, text, semantic):
    """
    __ -- (это) __
    """
    dash_ = max(text.lower().find('--'), text.lower().find('—'))

    if dash_ > -1:
        before = text[0:dash_-1]
        if text[0:5]=='Пусть':
            before = text[6:dash_ - 1]
        eto = max(text.lower().find('-- это'), text.lower().find('— это'))
        if eto > -1:
            after = text[dash_+6:]
        else:
            after = text[dash_+2:]
        return before, after
    return '0', '0'


operators_name = {'forall_', 'exist_', 'iff_'}
# Семантическая маска предложения
def give_semantic0(doc, sent):
    """
    Создает словать "слово" -- класс
    """
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
        elif word.lemma_ in iff:
            # определить место than
            semantic[i] = 'iff_'
        elif word.lemma_ in and_:
            # определить место than
            semantic[i] = 'and_'
        else:
            semantic[i] = '0'  #string
    return semantic


class MyObject:
    def __init__(self, type, text):
        self.Type = type
        self.Text = text
        if text == '':
            self.Text = self.Type


def get_lemma(nlp,word):
    doc = nlp(word)
    return doc[0].lemma_



def give_semantic(nlp, sent, semantic = [], temp_str = ''):
    """
    Создает массив my_objet-ов, содержащих информацию о классе слова/словосочетаия
    """
    doc = sent.split(' ')

    flag = 0
    for i in range(len(doc)):
        word = doc[i]
        #print(get_lemma(nlp, word))

        if flag:
            flag = 0

        elif get_lemma(nlp, word) in define:
            if temp_str != '':
                semantic.append(MyObject('str', temp_str))
            semantic.append(MyObject('is_', ''))
            temp_str = ''

        elif get_lemma(nlp, word) in foralls:
            if temp_str not in {'', ' для', 'Для'}:
                semantic.append(MyObject('str', temp_str))
            semantic.append(MyObject('forall_', ''))
            temp_str = ''

        elif get_lemma(nlp, word) in exists:
            if temp_str != '':
                semantic.append(MyObject('str', temp_str))
            semantic.append(MyObject('exist_', ''))
            temp_str = ''

        elif get_lemma(nlp, word) in iff:
            if temp_str != '':
                semantic.append(MyObject('str', temp_str))
                temp_str = ''
            idx = sent.find(word)

            left, right = if_then(sent[idx:], semantic)
            #print('left', left, 'right ', right)

            semantic.append(MyObject('iff_', ''))
            semantic, temp_str = give_semantic(nlp, left, semantic, temp_str)
            if temp_str != '':
                semantic.append(MyObject('str', temp_str))
                temp_str = ''

            semantic.append(MyObject('than_', ''))
            semantic, temp_str = give_semantic(nlp, right, semantic, temp_str)
            if temp_str != '':
                semantic.append(MyObject('str', temp_str))

            break

        elif (get_lemma(nlp, word) + ' ' + doc[min(i+1, len(doc)-1)]) in such:
            if temp_str != '':
                semantic.append(MyObject('str', temp_str))
            semantic.append(MyObject('such_', ''))
            temp_str = ''
            flag = 2

        else:
            temp_str = temp_str + ' ' + word  #string

            if i == len(doc) - 1:
                #print(temp_str)
                semantic.append(MyObject('str', temp_str))
                #temp_str = ''

    return semantic, temp_str


def no_operators(semantic):
    for s in semantic:
        if s.Type in {'forall_', 'exist_', 'is_', 'iff_', 'and_'}:
            return False
        if ' и ' in s.Text:
            return False
    return True

def build(semantic, tree):
    #pprint(semantic)
    prev_node = 0
    start_node = 0
    flag = 0
    for i, item in enumerate(semantic):
        if flag > 0:
            flag -=1
        elif item.Type in {'forall_', 'exist_'}:
            if ' и ' in semantic[i+1].Text:
                print('i')
                all_arg = semantic[i+1].Text
                and_pos = all_arg.find(' и ')
                left_arg = all_arg[0: and_pos]
                right_arg = all_arg[and_pos +3 :]
                arg_node1 = Node(left_arg)
                arg_node2 = Node(right_arg)
                node = Node(item.Type, suc=[arg_node2])
                node1 = Node(item.Type, suc=[arg_node1, node])
                if prev_node:
                    prev_node.suc.append(node1)
                else:
                    start_node = node1
                prev_node = node
                tree.extend([node1, node, arg_node1, arg_node2])
                flag = 1
            else:
                arg_node = Node(semantic[i+1].Text)
                node = Node(item.Type, suc=[arg_node])
                if prev_node:
                    prev_node.suc.append(node)
                else:
                    start_node = node
                prev_node = node
                tree.extend([node, arg_node])
                flag = 1

        elif item.Type == 'is_':
            arg1_node = Node(semantic[max(0, i-1)].Text)
            arg2_node = Node(semantic[min(i+1, len(semantic)-1)].Text)
            node = Node(item.Type, suc = [arg1_node, arg2_node])

            if prev_node:
                prev_node.suc.append(node)
            else:
                start_node = node

            tree.extend([node, arg1_node, arg2_node])
            prev_node = node
            flag = 1

        elif item.Type == 'iff_':
            end = len(semantic)
            for j in range(i, len(semantic)):
                if semantic[j].Type == 'than_':
                    end = j
                    break
            t, arg1_node = build(semantic[i+1:end], tree)

            # TODO: написать комментарий
            if end < len(semantic) - 1:
                t, arg2_node = build(semantic[end:len(semantic)], tree)
                node = Node('impl_', suc = [arg1_node, arg2_node])
                tree.extend([node, arg1_node, arg2_node])
            else:
                node = Node(item.Type, suc=[arg1_node])
                tree.extend([node, arg1_node])

            if prev_node:
                prev_node.suc.append(node)
            else:
                start_node = node

            return start_node, prev_node

        elif i == len(semantic) - 1:
            if ', ' in item.Text or ' и ' in item.Text:
                all_item = item.Text
                comma_pos = max(all_item.find(', '), all_item.find(' и '))
                left_arg = all_item[:comma_pos]
                right_arg = all_item[comma_pos+2 :]
                node1, node2 = Node(left_arg), Node(right_arg)
                node = Node('and_', suc =[node1, node2])
                tree.extend([node, node1, node2])
                if prev_node:
                    prev_node.suc.append(node)
                else:
                    start_node = node
                prev_node = node
            else:
                node = Node(item.Text)
                tree.extend([node])
                if prev_node:
                    prev_node.suc.append(node)
                else:
                    start_node = node
                prev_node = node

        else:
            pass
    return start_node, prev_node


def pprint(sem):
    for i in sem:
        print(i.Type, ':', i.Text)


# MAIN


def proccess(a_text, tree = [], draw = 0):
    eqv_templates = [define_if]
    impl_templates = [if_then]
    is_templates = [dash]
    nlp = spacy.load("ru_core_news_lg")
    for line in a_text:
        line, _ = replace_latex(line)
        sents = sent_tokenize(line, language='russian')
        for sent in sents:
            print('-----------------------------')
            print()
            print(sent)
            print()
            doc = nlp(sent)
            if draw:
                draw_syntax_tree(doc)
            semantic0 = give_semantic0(doc, sent)
            #print(semantic0)
            used_template = 0

            # корень-эквивалентность
            for template in eqv_templates:
                left, right = template(nlp, doc, sent, semantic0)

                if left != '0' and right != '0':
                    used_template = 1
                    print('l:', left)
                    print('r:', right)
                    left_semantic, _ = give_semantic(nlp, left, [])
                    right_semantic, _ = give_semantic(nlp, right, [])

                    left_child, _ = build(left_semantic, tree)
                    right_child, _ = build(right_semantic, tree)
                    eqv_root = Node('eqv', suc=[left_child, right_child])

                    tree.insert(0, eqv_root)

            # корень-импликация
            if not used_template:
                for template in impl_templates:
                    left, right = template(sent, semantic0)  # возвращать индексы или строки?
                    if left != '0' and right != '0':
                        used_template = 1
                        print('l:', left)
                        print('r:', right)
                        left_semantic, _ = give_semantic(nlp, left, [])
                        right_semantic,_ = give_semantic(nlp, right, [])

                        left_child, _ = build(left_semantic, tree)
                        right_child, _ = build(right_semantic, tree)
                        impl_root = Node('impl', suc = [left_child, right_child])
                        tree.insert(0, impl_root)



            if not used_template:
                for template in is_templates:

                    left, right = template(doc, sent, semantic0)

                    if left != '0' and right != '0':
                        used_template = 1
                        print('l:', left)
                        print('r:', right)
                        left_semantic, _ = give_semantic(nlp, left, [])
                        right_semantic, _ = give_semantic(nlp, right, [])

                        if no_operators(left_semantic):
                            left_child = Node(name = str(left))
                            tree.extend([left_child])
                        else:
                            left_child, _ = build(left_semantic, tree)

                        if no_operators(right_semantic):
                            right_child = Node(name = str(right))
                            tree.extend([right_child])
                        else:
                            right_child, _ = build(right_semantic, tree)

                        is_root = Node('is', suc=[left_child, right_child])
                        tree.insert(0, is_root)

            if not used_template:
                semantic, _ = give_semantic(nlp, sent, [])
                root, _ = build(semantic, tree)

            if len(tree):
                print_tree(tree[0])
            tree = []
    return tree



tree = proccess(examples)
