from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def bidirectional_search(problem):
    """
        Implement a bidirectional search algorithm that takes instances of SimpleSearchProblem (or its derived
        classes) and provides a valid and optimal path from the initial state to the goal state.

        :param problem: instance of SimpleSearchProblem
        :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
                 num_nodes_expanded: number of nodes expanded by your search
                 max_frontier_size: maximum frontier size during search
        """
    max_frontier_size = 0
    num_nodes_expanded = 0
    path = []

    # check case if we start at the goal
    if problem.init_state == problem.goal_states[0]:
        return [problem.init_state], 0, 1

    # initialization
    qforward = deque()  #FIFO queues for BFS, will store the nodes to be explored next
    qback = deque()
    visitedf = set()    #visited sets to keep track of all visited nodes
    visitedb = set()
    #to keep track of parents of each discovered node
    parentsf = {}   # key = current node 
    parentsb = {}   # value = parent of the key
    levelcount = 0  # current frontiers explored in each direction

    # adding the starting nodes to the queues
    qforward.append(problem.init_state)
    qback.append(problem.goal_states[0])
    # parentsf[problem.init_state] = [True, 0]
    # parentsb[problem.goal_states[0]] = [True, 0]

    # explore in forwards and backwards directions using bfs 1 level at a time
    while len(qforward)!=0 and len(qback)!=0:
        qforward, visitedf, parentsf, num_nodes_expanded, out = explorelevel(problem, qforward, visitedf, parentsf, num_nodes_expanded, qback, visitedb)
        qback, visitedb, parentsb, num_nodes_expanded, out1= explorelevel(problem, qback, visitedb, parentsb, num_nodes_expanded, qforward, visitedf)
        # print("queues - forward: ", qforward, "backward: ", qback)
        if out is not False:
            # the queues have intersected and we've found a solution
            path = findpath(parentsf, parentsb, out, problem)
            return path, num_nodes_expanded, max_frontier_size
        elif out1 is not False:
            path = findpath(parentsf, parentsb, out1, problem)
            return path, num_nodes_expanded, max_frontier_size

        levelcount = levelcount+1
    
    # check cases where there are still elements left in a queue
    while len(qforward)>0:
        # print("some left in forward")
        qforward, visitedf, parentsf, num_nodes_expanded, out = explorelevel(problem, qforward, visitedf, parentsf, num_nodes_expanded, qback, visitedb)
    if out is not False:
        # the queues have intersected and we've found a solution
        path = findpath(parentsf, parentsb, out.pop(), problem)
        return path, num_nodes_expanded, max_frontier_size
    while len(qback)>0:
        # print("some left in back")
        qback, visitedb, parentsb, num_nodes_expanded, out1= explorelevel(problem, qback, visitedb, parentsb, num_nodes_expanded, qforward, visitedf)
    if out is not False:
        # the queues have intersected and we've found a solution
        path = findpath(parentsf, parentsb, out.pop(), problem)
        return path, num_nodes_expanded, max_frontier_size

    # otherwise there is no path
    path = []
    max_frontier_size= levelcount*2
    return path, num_nodes_expanded, max_frontier_size

def findpath(p1, p2, mid, problem):
    """
        takes in the sets: forward parents p1, backward parents p2 (from the goal), 
        mid: node they meet at 
        problem: problem instance (GridSearchProblem)
        outputs: list of nodes representing the path
    """
    # print("lets get the path, which meets at", mid)
    path = []
    # goal to meeting node mid
    curr=mid
    # working from middle to the initial
    while curr != problem.init_state:
        path.append(p1[curr])
        curr = p1[curr]
    path = path[::-1] #reversing it
    
    # going the other direction for the second half
    curr = mid
    path.append(curr) # add the meeting node
    while curr != problem.goal_states[0]:
        path.append(p2[curr])
        curr=p2[curr]
    return path

def explorelevel(problem, q, v, p, n, q2, v2):
    """
        running bfs on the current level
        inputs: problem, the queue, the visited set, 
                the parents dictionary, number of nodes explored,
                from the other direction: the queue and visited lists
        outputs: q_out updated queue, v visited, p parents, and n number of nodes
                found = intersected node if it is found, False otherwise
    """
    q_out = deque()     #initializing output queue
    # will hold the next instance of the queue after exploring current level

    while len(q)!=0:
        #print("current level is", level)
        # want to check all of the other nodes in that level
        # all the nodes in the current level are all in the queue
        curr = q.popleft()      # update current node
        n= n+1                  # update number of nodes we've explored
        for i in problem.get_actions(curr):
            if i[1] in v: 
                #node has already been explored if it's in visited
                continue
            elif curr in v2 or curr in q2:
                #node is in the visited set of the other direction
                # or in the queue => intersection found
                return q_out, v, p, n, curr
            else:
                # else, node hasnt been discovered
                q_out.append(i[1])  # add it to the queue
                p[i[1]]= curr       # update its parent
                v.add(i[1])         # add the node into the visited set

    return q_out, v, p, n, False

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
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('./stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Be sure to compare with breadth_first_search!