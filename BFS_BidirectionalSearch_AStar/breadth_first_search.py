from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def breadth_first_search(problem):
    """
    Implement a simple breadth-first search algorithm that takes instances of SimpleSearchProblem (or its derived
    classes) and provides a valid and optimal path from the initial state to the goal state. Useful for testing your
    bidirectional and A* search algorithms.

    :param problem: instance of GraphSearchProblem
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    #check if we start at the goal
    if problem.init_state == problem.goal_states[0]:
        return [problem.init_state], 0, 1

    # initialization
    q = deque()     #FIFO queue for BFS, this will hold all of the nodes to be explored
    visited = set() #all visited nodes
                    # problem.V.shape[0] is the number of nodes
    max_frontier_size = 0   #return values
    num_nodes_expanded = 0
    path = []
    parents = {}  #parents will hold the previous node
        #key = current node, value = parent of the key      
    # initialize all parents to be False
    q.append(problem.init_state)#start by exploring the initial node

    while len(q)!=0:
        # keep looping until the queue is empty
        # occurs when there are no more nodes to explore from the initial
        curr = q.popleft() #current node to look at is the first element in the queue
            # curr is an integer! The current state
        if curr == problem.goal_states[0]:
            ## reached the goal node
            break
        # otherwise, we can look at all of its neighbors
        #get_actions: returns list of tuples corresponding to edges connected to state
        for i in problem.get_actions(curr): 
            #iterate through all the neighbors of current state
            if i[1] in visited:
                # if we have already explored the node before
                continue
            q.append(i[1])          # queue this unexplored node
            parents[i[1]] = curr    # set the parent of the node to be explored
            visited.add(i[1])       # append the nodes to the visited set

        #finished exploring the current node
        num_nodes_expanded = num_nodes_expanded+1
        
        # update max frontier size if necessary
        if len(q)>max_frontier_size:
            max_frontier_size= len(q)

    # check if we broke out of the loop since goal was reached
    if curr != problem.goal_states[0]:
        return [], num_nodes_expanded, max_frontier_size
    
    # work backwards from the goal to obtain path from the
    # parent nodes, then reverse path
    path.append(curr) 
    while curr != problem.init_state:
        #loop until we get to the first node again
        path.append(parents[curr])  #add parent to path
        curr = parents[curr]        #set current to be the parent
    path = path[::-1]               #reverse path when done
    
    return path, num_nodes_expanded, max_frontier_size


if __name__ == '__main__':
    # Simple example
    goal_states = [0]
    init_state = 9
    V = np.arange(0, 10)
    E = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [0, 6],
                  [1, 7],
                  [2, 5],
                  [9, 4]])
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('./stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)

    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)