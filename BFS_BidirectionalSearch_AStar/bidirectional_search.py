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

    # if we start at the goal
    if problem.init_state == problem.goal_states[0]:
        return [problem.init_state], 0, 1

    # initialization
    qforward = deque()  #FIFO queues for BFS, will store the nodes to be explored next
    qback = deque()
    visitedf = set()       #visited lists to keep track of all visited nodes
    visitedb = set()
    #to keep track of parents and level of each discovered node
    parentsf = {node:[False, -1] for node in problem.V} 
    parentsb = {node:[False, -1] for node in problem.V}
    levelcount = 0

    # adding the starting nodes to the queues
    qforward.append(problem.init_state)
    qback.append(problem.goal_states[0])
    parentsf[problem.init_state] = [True, 0]
    parentsb[problem.goal_states[0]] = [True, 0]
    # print("queues - forward: ", qforward, "backward: ", qback)
    # explore in forwards and backwards directions using bfs 1 level at a time
    while len(qforward)!=0 and len(qback)!=0:
        qforward, visitedf, parentsf, num_nodes_expanded = explorelevel(problem, qforward, visitedf, parentsf, num_nodes_expanded)
        qback, visitedb, parentsb, num_nodes_expanded = explorelevel(problem, qback, visitedb, parentsb, num_nodes_expanded)
        # print("queues - forward: ", qforward, "backward: ", qback)
        out, max_frontier_size = checkmet(qforward, qback, visitedf, visitedb, levelcount)
        # print("checking found: ", out)
        if out is not False:
            # the queues have intersected and we've found a solution
            path = findpath(parentsf, parentsb, out.pop(), problem)
            return path, num_nodes_expanded, max_frontier_size

        levelcount = levelcount+1
    
    # check cases where there are still elements left in a queue
    while len(qforward)>0:
        # print("some left in forward")
        qforward, visitedf, parentsf, num_nodes_expanded = explorelevel(problem, qforward, visitedf, parentsf, num_nodes_expanded)
    while len(qback)>0:
        # print("some left in back")
        qback, visitedb, parentsb, num_nodes_expanded = explorelevel(problem, qback, visitedb, parentsb, num_nodes_expanded)
    out, max_frontier_size = checkmet(qforward, qback, visitedf, visitedb, levelcount)
    if out is not False:
        # the queues have intersected and we've found a solution
        path = findpath(parentsf, parentsb, out.pop(), problem)
        return path, num_nodes_expanded, max_frontier_size

    # otherwise there is no path
    path = []
    max_frontier_size= levelcount*2
    return path, num_nodes_expanded, max_frontier_size

def checkmet(q1, q2, v1, v2, levels):
    # returns true when the 2 bfs searches intersect 
    # print("checking solution found")
    check = v1.intersection(set(q2))
    if len(check) > 0:
        return check, levels*2
    check = v2.intersection(set(q1))
    if len(check)>0:
        return check, levels*2
    check = set(q1).intersection(set(q2))
    if len(check)>0:
        return check, levels*2-1
    return False, -1

def findpath(p1, p2, mid, problem):
    # takes in the forward parents p1, backward parents p2 (from the goal), and the node they meet
    # outputs path list
    # print("lets get the path, which meets at", mid)
    path = []
    # goal to meeting node mid
    curr=mid
    # working from middle to the initial
    while curr != problem.init_state:
        # print(path)
        path.append(p1[curr][0])
        curr = p1[curr][0]
    path = path[::-1] #reversing it
    # print("halfway")
    curr = mid
    path.append(curr) # add the meeting node
    # print(p2[curr][0])
    while curr != problem.goal_states[0]:
        # print(path)
        path.append(p2[curr][0])
        curr=p2[curr][0]
    return path

def explorelevel(problem, q, v, p, n):
    # running bfs on the current level
    # the final queue 
    q_out, v_out, p_out, n_out = q, v, p, n
    # the queue, the visited set, the parents dictionary, and number of nodes
    #print("running explorelevel")
    curr = q_out.popleft() #get the first node in the queue
    level = p_out[curr][1] # level of the first node
    q_out.appendleft(curr) # append it back on so we can pop it in the loop

    while len(q_out)!=0:
        #print("current level is", level)
        # want to check all of the other nodes in that level 
        curr = q_out.popleft()      # update current node
        if p_out[curr][1] != level: # if we hit a new level, break
            break
        n_out= n_out+1
        for i in problem.get_actions(curr):
            if i[1] in v_out or i[1] in q_out: 
                #node has already been explored or is in the queue
                continue
            # else, node hasnt been discovered
            q_out.append(i[1]) # add it to the queue
            p_out[i[1]]=[curr, p_out[curr][1]+1] # update its parent and level
        v_out.add(curr)      # add the current node into the visited list
    
    if p_out[curr][1] > level: 
        # if the current node broke out of the loop since its on another frontier
        q_out.appendleft(curr) #add it back onto the queue

    return q_out, v_out, p_out, n_out

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