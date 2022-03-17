import re


def pprint_relation_statement(relation_statement: str) -> None:
    """
    prints relation statement with entity 1 colored as green, entity 2 colored as red
    and some persian token at first for correct RTL print
    :param relation_statement: relation statement text
    :return: None
    """
    p1 = re.sub(r'<e1>', '<e1>\033[92m', relation_statement)
    p2 = re.sub(r'</e1>', '\033[0m</e1>', p1)
    p3 = re.sub(r'<e2>', '<e2>\033[91m', p2)
    p4 = re.sub(r'</e2>', '\033[0m</e2>', p3)
    print(f'جمله: {p4}')
