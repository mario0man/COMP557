import collections, util, math, random

############################################################
# Problem 3.1.1

def computeQ(mdp, V, state, action):
    """
    Return Q(state, action) based on V(state).  Use the properties of the
    provided MDP to access the discount, transition probabilities, etc.
    In particular, MDP.succAndProbReward() will be useful (see util.py for
    documentation).  Note that |V| is a dictionary.  
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    # raise Exception("Not implemented yet")
    sum = 0
    successors = mdp.succAndProbReward(state, action)
    for successor in successors:
        newState, prob, reward = successor
        sum += prob * (reward + mdp.discount() * V[newState])
    return sum
    # END_YOUR_CODE

############################################################
# Problem 3.1.2

def policyEvaluation(mdp, V, pi, epsilon=0.001):
    """
    Return the value of the policy |pi| up to error tolerance |epsilon|.
    Initialize the computation with |V|.  Note that |V| and |pi| are
    dictionaries.
    """
    # BEGIN_YOUR_CODE (around 8 lines of code expected)
    # raise Exception("Not implemented yet")
    mdp.computeStates()
    states = mdp.states 
    V = {state:0 for state in states}
    while True: 
        V2 = {state:computeQ(mdp, V, state, pi[state]) for state in states} 
        diff = [0 if abs(V2[state] - V[state]) <= epsilon else 1 for state in states]
        if sum(diff) == 0:
            break
        V = V2 
    return V 
    # END_YOUR_CODE

############################################################
# Problem 3.1.3

def computeOptimalPolicy(mdp, V):
    """
    Return the optimal policy based on V(state).
    You might find it handy to call computeQ().  Note that |V| is a
    dictionary.
    """
    # BEGIN_YOUR_CODE (around 4 lines of code expected)
    # raise Exception("Not implemented yet")
    pi = {}
    for state in mdp.states:
        max_q = -99999
        best_a = None
        for a in mdp.actions(state):
            qval = computeQ(mdp, V, state, a)
            if qval > max_q:
                max_q = qval
                best_a = a
        pi[state] = a
    return pi
    # END_YOUR_CODE
############################################################
# Problem 3.1.4

class PolicyIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        # compute |V| and |pi|, which should both be dicts
        # BEGIN_YOUR_CODE (around 8 lines of code expected)
        V = {}
        pi = {}
        for s in mdp.states:
            V[s] = 0
            pi[s] = mdp.actions(s)[0]

        while(True):
            Vnew = policyEvaluation(mdp, V, pi, epsilon)
            pinew = computeOptimalPolicy(mdp, Vnew)
            if(pi == pinew):
                V = Vnew
                break;
            pi = pinew
            V = Vnew

        # END_YOUR_CODE
        self.pi = pi
        self.V = V

############################################################
# Problem 3.1.5

class ValueIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        states = mdp.states
        # BEGIN_YOUR_CODE (around 11lines of code expected)
        V = {}
        pi = {}
        for s in states:
            V[s] = 0
            pi[s] = 1

        while(True):
            Vnew = {}
            flag = True
            for s in states:
                max_q = -999999999
                max_a = 1
                for a in mdp.actions(s):
                    qval = computeQ(mdp, V, s, a)
                    if qval > max_q:
                        max_q = qval
                        max_a = a

                Vnew[s] = max_q
                pi[s] = max_a

                if abs(Vnew[s] - V[s]) > epsilon:
                    flag = False
            if flag:
                V = Vnew
                break;

            V = Vnew
        # END_YOUR_CODE
        self.pi = pi
        self.V = V
############################################################
# Problem 3.1.6

# If you decide 1f is true, prove it in writeup.pdf and put "return None" for
# the code blocks below.  If you decide that 1f is false, construct a
# counterexample by filling out this class and returning an alpha value in
# counterexampleAlpha().
class CounterexampleMDP(util.MDP):
    def __init__(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        # raise Exception("Not implemented yet")
        return 
        # END_YOUR_CODE

    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        # raise Exception("Not implemented yet")
        return 0
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        # raise Exception("Not implemented yet")
        return ['left']
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        # raise Exception("Not implemented yet")
        if state == 0: return [(1, .4, 10), (2, .6, 0)]
        return []
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        # raise Exception("Not implemented yet")
        return 1
        # END_YOUR_CODE

def counterexampleAlpha():
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    # raise Exception("Not implemented yet")
    return .5
    # END_YOUR_CODE

############################################################
# Problem 3.2.1

class StateClass:
    def __init__(self, totalScore=None, nextCard=None, deckState=None):
        self.totalScore = totalScore
        self.nextCard = nextCard
        self.deckState = deckState

    def fromTuple(self, tup):
        self.totalScore, self.nextCard, self.deckState = tup
        return self 

    def toTuple(self):
        return (self.totalScore, self.nextCard, self.deckState)

    def removeCard(self, card):
        temp = list(self.deckState)
        temp[card] -= 1
        self.deckState = tuple(temp)

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.  The second element is the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.  The final element
    # is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to (0,).
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 40 lines of code expected)
        # raise Exception("Not implemented yet")
        totalScore, nextCard, deckState = state
        if sum(deckState) == 0: # if no cards in deck, no next moves
            return [] 

        if action == 'Quit': # if quit, collect current score as reward
            return [((totalScore, None, (0,)), 1, totalScore)]

        if action == 'Peek':
            nextMoves = []
            if nextCard is None: #if peeking happened once, pick guaranteed card
                for card in range(len(deckState)):
                    nextState = (totalScore, card, deckState)
                    pr = deckState[card]/float(sum(deckState))
                    nextMoves.append((nextState, pr, -self.peekCost))
            return nextMoves

        if action == "Take":
            nextMoves = [] 
            if nextCard is None: # if no card found via peeking, pick random
                for card in range(len(deckState)):
                    nextValue = self.cardValues[card]
                    pr = deckState[card]/float(sum(deckState))
                    if pr == 0: # move to next card if no prob to pick this one
                        continue 
                    if totalScore + nextValue > self.threshold: # bust-> end game
                        nextMoves.append(((totalScore + nextValue, None, (0,)), 
                            pr, 0))
                    else: # not bust -> keep playing 
                        temp = StateClass().fromTuple(state)
                        temp.removeCard(card)
                        _, _, nextDeck = temp.toTuple()
                        nextMoves.append(((totalScore + nextValue, None, nextDeck), pr, 0))
                if sum(deckState) == 1: # if only 1 card left, game end and reward = curr score
                    return [((nextMoves[0][0][0], None, nextMoves[0][0][2]), 1, nextMoves[0][0][0])]
                return nextMoves
            else: # if next card already found via peeking, take that card
                temp = StateClass().fromTuple(state)
                temp.removeCard(state[1])
                _, _, nextDeck = temp.toTuple()
                return [((totalScore + nextCard, None, nextDeck), 1, 0)]
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3.2.2

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    # raise Exception("Not implemented yet")
    return BlackjackMDP([10, 20], 4, 20, 1)
    # END_YOUR_CODE

# def main():
#     a = StateClass(1, 2, 3)
#     print a
#     b = a.toTuple()
#     print b
#     c = StateClass().fromTuple((1,2,(4,5)))
#     c.removeCard(1)
#     print c.deckState

# # main() 