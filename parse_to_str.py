class Node:
    def __init__(self, name, i=0, suc=None, syntax=None, head=0):
        self.name = name
        self.suc = suc if suc is not None else []
    def __repr__(self):
        return f"{self.name} {self.suc}" #if self.suc else f"{self.name}"


def display_as_expression(root) -> str:

    if not root.suc:
        return root.name

    if root.name in {'forall_', 'exist_'}:
        assert len(root.suc) == 2
        expr = display_as_expression(root.suc[1])
        return f'{root.name} {root.suc[0].name} {expr}'
    else:
        args = ','.join(map(display_as_expression, root.suc))
        return f'root.name({args})'
