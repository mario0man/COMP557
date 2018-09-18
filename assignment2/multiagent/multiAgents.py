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

class ReflexAgent(Agent)	:
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
        def Vopt(gameState, d, agent, eval):
            TotalAgents = gameState.getNumAgents()
            LegalActions = gameState.getLegalActions()
            MaxDepth = self.depth
            PLAYER1 = 0

            if(agent >= TotalAgents):
                d += 1 
                agent = PLAYER1

            if(gameState.isWin() or gameState.isLose() or d == MaxDepth):
                return eval(gameState)

            if(agent == 0):
                #Agent is the Player
                actions = gameState.getLegalActions(agent)
                return MaxLayer(gameState, d, agent, actions, eval)

            if(agent > 0):
                #Agent is a Ghost
                actions = gameState.getLegalActions(agent)
                return MinLayer(gameState, d, agent, actions, eval)

        def MaxLayer(gameState, d, agent, actions, eval):

            if len(actions) == 0:
                return eval(gameState)

            BestMove = ("South", -999999999999)

            for a in actions:
                NextState = gameState.generateSuccessor(agent, a)
                NewAgent = agent + 1
                UtilityValue = Vopt(NextState, d, NewAgent, eval)

                if type(UtilityValue) is not float:
                    UtilityValue = UtilityValue[1]

                if UtilityValue > BestMove[1]:
                    BestMove = [a, UtilityValue]

            return BestMove

        def MinLayer(gameState, d, agent, actions, eval):

            if len(actions) == 0:
                return eval(gameState)

            BestMove = ("South", 99999999999)

            for a in actions:
                NextState = gameState.generateSuccessor(agent, a)
                NewAgent = agent + 1
                UtilityValue = Vopt(NextState, d, NewAgent, eval)

                if type(UtilityValue) is not float:
                    UtilityValue = UtilityValue[1]

                if UtilityValue < BestMove[1]:
                    BestMove = [a, UtilityValue]

            return BestMove

        return Vopt(gameState, 0, 0, self.evaluationFunction)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        def Vopt(gameState, d, agent, eval, alpha, beta):
            TotalAgents = gameState.getNumAgents()
            LegalActions = gameState.getLegalActions()
            MaxDepth = self.depth
            PLAYER1 = 0

            if(agent >= TotalAgents):
                d = d+1
                agent = PLAYER1

            if(gameState.isWin() or gameState.isLose() or d == MaxDepth):
                return eval(gameState)

            if(agent == 0):
                #Agent is the Player
                actions = gameState.getLegalActions(agent)
                return MaxLayer(gameState, d, agent, actions, eval, alpha, beta)

            if(agent > 0):
                #Agent is a Ghost
                actions = gameState.getLegalActions(agent)
                return MinLayer(gameState, d, agent, actions, eval, alpha, beta)

        def MaxLayer(gameState, d, agent, actions, eval, alpha, beta):

            if len(actions) == 0:
                return eval(gameState)

            BestMove = ("South", -999999999999)
            TotalAgents = gameState.getNumAgents()

            for a in actions:
                NextState = gameState.generateSuccessor(agent, a)
                NewAgent = agent + 1
                UtilityValue = Vopt(NextState, d, NewAgent, eval, alpha, beta)

                if type(UtilityValue) is not float:
                    UtilityValue = UtilityValue[1]

                if UtilityValue > BestMove[1]:
                    BestMove = [a, UtilityValue]

                if UtilityValue > beta: 
                	return BestMove

                alpha = max(alpha, UtilityValue)

            return BestMove

        def MinLayer(gameState, d, agent, actions, eval, alpha, beta):

            if len(actions) == 0:
                return eval(gameState)

            BestMove = ("South", 99999999999)
            TotalAgents = gameState.getNumAgents()

            for a in actions:
                NextState = gameState.generateSuccessor(agent, a)
                NewAgent = agent + 1
                UtilityValue = Vopt(NextState, d, NewAgent, eval, alpha, beta)

                if type(UtilityValue) is not float:
                    UtilityValue = UtilityValue[1]

                if UtilityValue < BestMove[1]:
                    BestMove = [a, UtilityValue]

                if UtilityValue < alpha:
                	return BestMove

                beta = min(beta, UtilityValue)

            return BestMove

        return Vopt(gameState, 0, 0, self.evaluationFunction, -float('inf'), float('inf'))[0]

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

        def Vopt(gameState, d, agent, eval):
            TotalAgents = gameState.getNumAgents()
            LegalActions = gameState.getLegalActions()
            MaxDepth = self.depth
            PLAYER1 = 0

            if(agent >= TotalAgents):
                d = d+1
                agent = PLAYER1

            if(gameState.isWin() or gameState.isLose() or d == MaxDepth):
                return eval(gameState)

            if(agent == 0):
                #Agent is the Player
                actions = gameState.getLegalActions(agent)
                return MaxLayer(gameState, d, agent, actions, eval)

            if(agent > 0):
                #Agent is a Ghost
                actions = gameState.getLegalActions(agent)
                return ChanceLayer(gameState, d, agent, actions, eval)

        def MaxLayer(gameState, d, agent, actions, eval):

            if len(actions) == 0:
                return eval(gameState)

            BestMove = ("South", -999999999999)
            TotalAgents = gameState.getNumAgents()

            for a in actions:
                NextState = gameState.generateSuccessor(agent, a)
                NewAgent = agent + 1
                UtilityValue = Vopt(NextState, d, NewAgent, eval)

                if type(UtilityValue) is not float:
                    UtilityValue = UtilityValue[1]

                if UtilityValue > BestMove[1]:
                    BestMove = [a, UtilityValue]

            return BestMove

        def ChanceLayer(gameState, d, agent, actions, eval):
            ExpectedValue = 0
            Action = "South"
            TotalAgents = gameState.getNumAgents()


            if len(actions) == 0:
                return eval(gameState)

            for a in actions:
                NextState = gameState.generateSuccessor(agent, a)
                NewAgent = agent + 1
                UtilityValue = Vopt(NextState, d, NewAgent, eval)

                if type(UtilityValue) is not float:
                    UtilityValue = UtilityValue[1]

                ExpectedValue += (UtilityValue * (1/float(len(actions))))

            return [random.choice(actions), ExpectedValue]

        return Vopt(gameState, 0, 0, self.evaluationFunction)[0]


def betterEvaluationFunction(currentGameState):
    '''
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: This Evaluation Function looks at 3 information concerning
      the board state.

      First one is the current Game Score. If the game score is higher, it means
      we are deeper down the game tree, most of the time implying that we are on
      a desired game state. Higher score is desired for a state.

      Second information is the food factor. We calculate the distance to the food
      that is closest to pacman. We take its reciprocal, and add it to the
      score. So if there is a food that is really close, the score of the state
      will increase drastically. This might be considered a greedy strategy, since
      it might create a lot of risk to be eaten by the ghosts, just to get food.

      Third information is the distance to closest ghost. Originally we were
      going to decrease the score, if the ghost was close. But it resulted in a
      lot of losses. Just by experimenting we realized that the opposite was true.
      If you watch it with the graphics on, pacman actually follows the ghosts, and
      for some reason it's giving him an advantage. It might be because since the ghost
      is close, we have more predictive power of where it will be after a couple of
      moves later.
    # Useful information you can extract from a GameState (pacman.py)

    Increase score value if:
    - far from ghosts without power capsule powerup
    - close to ghost with power capsule powerup
    - close to power capsule
    - close to food
    '''
    # return successorGameState.getScore()
    FoodPositions = currentGameState.getFood().asList()
    FoodDistances = []
    GhostStates = currentGameState.getGhostStates()
    GhostDistances = []
    CapsulePositions = currentGameState.getCapsules()
    CurrentPosition = list(currentGameState.getPacmanPosition())
    ghostFactor = 1
    foodFactor = 1


    for f in FoodPositions:
        d = manhattanDistance(f, CurrentPosition)
        FoodDistances.append(d)

    if len(FoodDistances) == 0:
        FoodDistances = [0]

    foodFactor = min(FoodDistances)

    for ghost in GhostStates:
    	d = manhattanDistance(CurrentPosition, ghost.getPosition())
        GhostDistances.append(d)

    if len(GhostDistances) == 0:
        GhostDistances = [99999]

    ghostFactor = min(GhostDistances)

    if ghostFactor == 0: ghostFactor = 1
    if foodFactor == 0: foodFactor = 1

    #return currentGameState.getScore() + (1.0/foodFactor) + (1.0/ghostFactor)
    return currentGameState.getScore() + (1.0/foodFactor) - ghostFactor

# Abbreviation
better = betterEvaluationFunction
