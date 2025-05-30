
def print_tree(node, prefix="", is_last=True, is_root=True):
    """
    Рекурсивно выводит дерево с отступами и символами ветвей.
    """
    if is_root:
        print(node.name)
    else:
        print(prefix + ("└── " if is_last else "├── ") + node.name)

    # Формируем новый префикс для дочерних элементов
    new_prefix = prefix + ("    " if is_last else "│   ")

    # Рекурсивно обрабатываем дочерние узлы
    for i, child in enumerate(node.suc):
        is_last_child = (i == len(node.suc) - 1)
        print_tree(child, new_prefix, is_last_child, is_root=False)



