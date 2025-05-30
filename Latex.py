import networkx as nx
import matplotlib.pyplot as plt
import re
from typing import *

def replace_latex(text):
    """
    Производит замену всех latex-формул на символ `+n`.
    """
    # Регулярное выражение для поиска подстрок между `$`:
    pattern = r'\$(.*?)\$'

    counter: int = 0
    mapping: Dict[int, str] = {}

    def make_sign(a_formula) -> str:
        nonlocal counter, mapping
        counter += 1
        # print('>>', dir(a_formula))
        # print('>>>>', a_formula.group(0))
        mapping[counter] = a_formula.group(0)
        return "+"+f'{counter}'


    res = re.sub(pattern, make_sign, text)

    return res, mapping