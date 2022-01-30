from collections import deque
import queue
import numpy as np
from search_problems import Node, GridSearchProblem, get_random_grid_problem

def a_star_search(problem):
    """
    Uses the A* algorithm to solve an instance of GridSearchProblem. Use the methods of GridSearchProblem along with
    structures and functions from the allowed imports (see above) to implement A*.

    :param problem: an instance of GridSearchProblem to solve
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    if problem.init_state == problem.goal_states[0]:
        # case where the end is the beginning
        return [problem.init_state], 0, 1
    
    # Initialization
    num_nodes_expanded = 0
    max_frontier_size = 0
    visited = set()               # consists of nodes that have been explored
    q = queue.PriorityQueue()     # priority queue with nodes to explore next
        # each element in the queue is a tuple (priority, node)
    parents = {}      # key= node, value= [predecessor, associated cost = g+h]

    # starting with the initial node
    q.put((0, problem.init_state))
    parents[problem.init_state] = [False, 0]

    while q.empty() is False:
        # while there are nodes in the queue to be explored
        curr= q.get()
        if curr == problem.goal_states[0]:
            # found the goal
            break
        for i in problem.get_actions(curr):
            # calculate its A* cost = (path cost of its parent + 1) + heuristic function
            cost = parents[curr][1]+1 + problem.heuristic(i[1])
            if i[1] in parents:
                # if it has been in parents, update it to have the minimum cost parent
                if cost < parents[i[1]][1]:
                    # current cost is less than previous
                    parents[i[1]]= [curr, cost]
                # otherwise, its cost is higher and we go with the previous parent
            
            if i[1] in visited:
                # we have already explored the node
                continue
            q.put((parents[i[1]][1], i[1]))    # add the unexplored node to the queue
        
        # we have fully explored the current node
        visited.add(curr)                   # add current node to visited
        num_nodes_expanded+=1               # increment # of nodes expanded
        if q.qsize() > max_frontier_size:   # update maximum frontier size if needed
            max_frontier_size= q.qsize()

    # if outside of the loop and we haven't met the goal
    if curr != problem.goal_states[0]:
        return [], num_nodes_expanded, max_frontier_size
    
    # otherwise, we have met the goal
    # work backwards from the goal to obtain path from the
    # parent nodes, then reverse path
    path = []
    path.append(curr) 
    while curr != problem.init_state:
        #loop until we get to the first node again
        path.append(parents[curr][0])  #add parent to path
        curr = parents[curr][0]        #set current to be the parent
    path = path[::-1]                  #reverse path when done

    return path, num_nodes_expanded, max_frontier_size


def search_phase_transition():
    """
    Simply fill in the prob. of occupancy values for the 'phase transition' and peak nodes expanded within 0.05. You do
    NOT need to submit your code that determines the values here: that should be computed on your own machine. Simply
    fill in the values!

    :return: tuple containing (transition_start_probability, transition_end_probability, peak_probability)
    """
    ####
    #   REPLACE THESE VALUES
    ####
    transition_start_probability = -1.0
    transition_end_probability = -1.0
    peak_nodes_expanded_probability = -1.0
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability


if __name__ == '__main__':
    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.25
    M = 10
    N = 10
    problem = get_random_grid_problem(p_occ, M, N)
    print(problem.get_actions(1))
    # Solve it
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))
    # Plot the result
    problem.plot_solution(path)

    # Experiment and compare with BFS