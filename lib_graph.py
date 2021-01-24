def dfs_paths(graph, start): 
    '''
    #Поиск путей обхода графа (вершины графа это линии, ребра это пересечение линии с другими линиями)
    '''
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        for next in set(graph[vertex]) - set(path):
            if len(path + [next]) == 4: #Нас интересуют все пути с 4 вершинами
                #Проверим существует ли пересечение первой и последней вершины, если да, включим результат в ответ
                if graph[next].count(path[0]) > 0:
                    yield path + [next]
                # else:
                #     print("not!!!")    
            else:
                stack.append((next, path + [next]))

def in_dictionary(key, dict):
    '''
    #Функция возвращает есть ли ключ в заданном словаре
    '''

    if key in dict:
        return True
    return False

def del_graph_vertex(graph, vertex):
    '''
    #Функция удаляет вершину графа
    '''

    graph.pop(vertex)
    for idx in graph:
        if graph[idx].count(vertex) > 0:
            graph[idx].remove(vertex)