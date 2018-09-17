# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        '''
        Increase score value if: 
        - far from ghosts without power capsule powerup
        - close to ghost with power capsule powerup
        - close to power capsule 
        - close to food 
        '''
        # return successorGameState.getScore()
        oldCapsules = currentGameState.getCapsules() 
        newCapsules = successorGameState.getCapsules() 
        capsuleWasEaten = (len(oldCapsules) < len(newCapsules))

        ghostFactor, capsuleFactor, foodFactor = 0.0, 0.0, 0.0

        for ghost in newGhostStates:
        	d = manhattanDistance(newPos, ghost.getPosition())
        	if d < 2: 
	        	if ghost.scaredTimer > 0: 
	        		ghostFactor += 1000
	        	else: 
	        		ghostFactor -= 1000

        for capsule in oldCapsules: 
        	d = manhattanDistance(newPos, capsule) 
        	capsuleFactor += 1/(0.1 + d) 

        for food in currentGameState.getFood().asList():
        	d = manhattanDistance(newPos, food)
        	foodFactor += 1/(0.1 + d*d)


        return ghostFactor + capsuleFactor + foodFactor


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def value(self, state, agentIndex): 
        	if depth or gameState.isWin() or gameState.isLose():
        		return scoreEvaluationFunction(state)

        	if agentIndex = 0: 
        		return maxValue(state, agentIndex, depth) 


  #       v = float("-inf")
  #       bestAction = []
  #       agent = 0
  #       actions = gameState.getLegalActions(agent)
  #       successors = [(action, gameState.generateSuccessor(agent, action)) for action in actions]
  #       for successor in successors:
  #           temp = self.minimax(1, range(gameState.getNumAgents()), successor[1], self.depth, self.evaluationFunction)
  #           if temp > v:
  #             v = temp
  #             bestAction = successor[0]
  #       return bestAction
  #   def minimax(self, agent, agentList, state, depth, evalFunc):
		# # at Leaf
		# # print agent, depth
		# if depth <= 0 or state.isWin() == True or state.isLose() == True:
		# 	return evalFunc(state)

		# if agent == 0:
		# 	v = float("-inf")
		# else:
		# 	v = float("inf")
		      
		# actions = state.getLegalActions(agent)
		# successors = [state.generateSuccessor(agent, action) for action in actions]
		# for j in range(len(successors)):
		# 	successor = successors[j]
		# 	if agent == 0: 
		# 	  v = max(v, self.minimax(agentList[agent+1], agentList, successor, depth, evalFunc))
		# 	elif agent == agentList[-1]:
		# 	  v = min(v, self.minimax(agentList[0], agentList, successor, depth - 1, evalFunc))
		# 	else:
		# 	  v = min(v, self.minimax(agentList[agent+1], agentList, successor, depth, evalFunc))
		# return v    


        # util.raiseNotDefined()
    #     bestScore, bestMove = self.maxFunction(gameState,self.depth)

    #     return bestMove

    # def maxFunction(self,gameState,depth):
    #     if depth==0 or gameState.isWin() or gameState.isLose():
    #       return self.evaluationFunction(gameState), "noMove"

    #     moves=gameState.getLegalActions()
    #     scores = [self.minFunction(gameState.generateSuccessor(self.index,move),1, depth) for move in moves]
    #     bestScore=max(scores)
    #     bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    #     chosenIndex = bestIndices[0]
    #     return bestScore,moves[chosenIndex]

    # def minFunction(self,gameState,agent, depth):  
    #     if depth==0 or gameState.isWin() or gameState.isLose():
    #       return self.evaluationFunction(gameState), "noMove"
    #     moves=gameState.getLegalActions(agent) #get legal actions.
    #     scores=[]
    #     if(agent!=gameState.getNumAgents()-1):
    #       scores =[self.minFunction(gameState.generateSuccessor(agent,move),agent+1,depth) for move in moves]
    #     else:
    #       scores =[self.maxFunction(gameState.generateSuccessor(agent,move),(depth-1))[0] for move in moves]
    #     minScore=min(scores)
    #     worstIndices = [index for index in range(len(scores)) if scores[index] == minScore]
    #     chosenIndex = worstIndices[0]
    #     return minScore, moves[chosenIndex]	

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

