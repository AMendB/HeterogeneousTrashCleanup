import numpy as np
import heapq

import numpy as np
import heapq

def heuristic(a, b):
    """Función heurística: Distancia de Manhattan"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_find_path(mapa, start, goal, return_distance=False):
    """Algoritmo A* para encontrar el camino más corto en un mapa."""
    # Direcciones posibles: arriba, abajo, izquierda, derecha
    vecinos = [(0, 1), (1, 0), (0, -1), (-1, 0),
               (1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    # Inicializar la cola de prioridad con el nodo de inicio
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    # Diccionario para almacenar el costo acumulado desde el inicio
    g_score = {start: 0}
    
    # Diccionario para almacenar el camino
    came_from = {}
    
    while open_set:
        # Sacar el nodo con el menor costo estimado
        current = heapq.heappop(open_set)[1]
        
        # Si llegamos al objetivo, reconstruir el camino y devolver la distancia
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            if return_distance:
                return len(path) - 1  # La distancia es el número de pasos
            else:
                return path
        
        # Explorar los vecinos
        for dx, dy in vecinos:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Verificar si el vecino está dentro del mapa y es navegable
            if (0 <= neighbor[0] < mapa.shape[0] and 
                0 <= neighbor[1] < mapa.shape[1] and 
                mapa[neighbor[0], neighbor[1]] == 1):
                
                # Calcular el costo acumulado hasta el vecino
                tentative_g_score = g_score[current] + 1
                
                # Si el vecino no ha sido visitado o encontramos un camino mejor
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
    
    # Si no se encuentra un camino, devolver -1
    return -1

# Ejemplo de uso
mapa = np.array([
    [1, 1, 1, 1, 0],
    [1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1]
])

if __name__ == '__main__':
    scenario_map = np.genfromtxt(f'Environment/Maps/challenging_map_big.csv', delimiter=',')
    start = (10, 10)
    goal = (62, 13)

    distancia = a_star_find_path(scenario_map, start, goal, return_distance=True)
    print(f"Distancia entre {start} y {goal}: {distancia}")

    path = a_star_find_path(scenario_map, start, goal)

    # Visualizar el mapa y el camino encontrado
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(scenario_map, cmap='binary')

    if path:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=3, label='Path')
        plt.plot(path[0, 1], path[0, 0], 'go', markersize=15, label='Start')
        plt.plot(path[-1, 1], path[-1, 0], 'ro', markersize=15, label='Goal')

    plt.grid(True)
    plt.legend(fontsize=12)
    plt.title("A* Pathfinding Result")
    plt.show()