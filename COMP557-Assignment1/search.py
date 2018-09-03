# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """

    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    frontier = util.Stack()

    visited = set() 

    start = problem.getStartState()
    print "Start:", start 

    frontier.push((start, []))

    while not frontier.isEmpty():
      current = frontier.pop()
      #print "current:", current
      currCoordinate = current[0]
      #print "currCoordinate:", currCoordinate
      path = current[1]
      #print "path:", path

      if problem.isGoalState(currCoordinate):
          return path
      if currCoordinate not in visited:
          visited.add(currCoordinate)
          for nbr in problem.getSuccessors(currCoordinate):
              if nbr[0] not in visited:
                    #print "succ[1]:", succs[1]
                    frontier.push((nbr[0], path + [nbr[1]]))

                    """if succs not in frontier:
                        frontier.push((succs[0], path + [succs[1]]))
                    else: 
                        del frontier[succs]
                        """""
    """
    frontier = util.Stack()
    visited = []
    print "Start:", problem.getStartState()
    frontier.push(problem.getStartState())
    while frontier:
        current = frontier.pop()
        print "Current:", current
        visited.append(current)
        if(problem.isGoalState(current)):
            return True
        else:
            succs = problem.getSuccessors(current)
            for successor1, action1, cost1 in succs:
                if not successor1 in visited:
                    frontier.push(successor1)
    return false
"""


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    frontier = util.Queue()

    visited = set() 

    start = problem.getStartState()
    #print "Start:", start 

    frontier.push((start, []))

    while not frontier.isEmpty():
      current = frontier.pop()
      #print "current:", current
      currCoordinate = current[0]
      #print "currCoordinate:", currCoordinate
      path = current[1]
      #print "path:", path

      if problem.isGoalState(currCoordinate):
          return path
      if currCoordinate not in visited:
          visited.add(currCoordinate)
          for nbr in problem.getSuccessors(currCoordinate):
              if nbr[0] not in visited:
                    #print "succ[1]:", succs[1]
                    frontier.push((nbr[0], path + [nbr[1]]))

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()

    start = problem.getStartState()
    #print "Start:", start 

    frontier.push((start, []), 0)

    visited = set() 

    while not frontier.isEmpty():
      current = frontier.pop()
      #print "current:", current
      currCoordinate = current[0]
      #print "currCoordinate:", currCoordinate
      path = current[1]
      #print "path:", path

      if problem.isGoalState(currCoordinate):
          return path
      if currCoordinate not in visited:
          visited.add(currCoordinate)
          for nbr in problem.getSuccessors(currCoordinate):
              if nbr[0] not in visited:
                cost = problem.getCostOfActions(path + [nbr[1]])
                #print "cost:", cost
                frontier.push((nbr[0], path + [nbr[1]]),cost)
                """tempFrontier = frontier
                flag = False
                while not tempFrontier.isEmpty():
                    temp = tempFrontier.pop()
                    if nbr [0] == temp[0]:
                        flag = True
                        break
                cost = problem.getCostOfActions(path + [succs[1]])
                if not flag:
                    frontier.push((nbr[0], path + [nbr[1]]),cost)
                else: 
                    if temp [1] < cost:
                        """


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()

    start = problem.getStartState()
    #print "Start:", start 

    frontier.push((start, []), 0 + heuristic(start,problem))
    print "heuristic(start,problem):", heuristic(start,problem)

    visited = set() 

    while not frontier.isEmpty():
      current = frontier.pop()
      #print "current:", current
      currCoordinate = current[0]
      #print "currCoordinate:", currCoordinate
      path = current[1]
      #print "path:", path

      if problem.isGoalState(currCoordinate):
          return path
      if currCoordinate not in visited:
          visited.add(currCoordinate)
          for nbr in problem.getSuccessors(currCoordinate):
              if nbr[0] not in visited:
                cost = problem.getCostOfActions(path + [nbr[1]]) 
                hCost = heuristic(nbr[0],problem) #- heuristic(current[0],problem)
                #print "cost:", cost
                frontier.push((nbr[0], path + [nbr[1]]),cost + hCost)
                #print "(heuristic(nbr[0],problem): ", heuristic(nbr[0],problem)
                #print "heuristic(current[0],problem): ", heuristic(current[0],problem)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
