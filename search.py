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
from util import Stack, Queue

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

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
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

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # stack_state_path: ((state, path), [visited_states])
    stack_state_path = Stack()

    visited_states = []  # Visited states

    # Check if the initial state is the goal state #
    if problem.isGoalState(problem.getStartState()):
        return []

    # Start from the beginning and find a solution, the path is an empty list #
    stack_state_path.push((problem.getStartState(), []))

    while True:
        # Terminate condition: can't find a solution #
        if stack_state_path.isEmpty():
            return []

        # Get information of the current state #
        current_state, current_path = stack_state_path.pop()
        visited_states.append(current_state)

        # Terminate condition: reach the goal #
        if problem.isGoalState(current_state):
            return current_path

        # Get successors of the current state #
        successors = problem.get_successors(current_state)

        # Add new states to the stack and update their path #
        if successors:
            for successor in successors:
                successor_state, action, _ = successor

                if successor_state not in visited_states:
                    new_path = current_path + [action]  # Calculate the new path
                    stack_state_path.push((successor_state, new_path))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # queueXY: ((x,y),[path]) #
    queue_state_path = Queue()

    visited_states = []  # Visited states

    # Check if the initial state is the goal state #
    if problem.isGoalState(problem.getStartState()):
        return []

    # Start from the beginning and find a solution, the path is an empty list #
    queue_state_path.push((problem.getStartState(), []))

    while True:
        # Terminate condition: can't find a solution #
        if queue_state_path.isEmpty():
            return []

        # Get information of the current state #
        current_state, current_path = queue_state_path.pop()
        visited_states.append(current_state)

        # Terminate condition: reach the goal #
        if problem.isGoalState(current_state):
            return current_path

        # Get successors of the current state #
        successors = problem.get_successors(current_state)

        # Add new states to the queue and update their path #
        if successors:
            for successor in successors:
                successor_state, action, _ = successor

                if successor_state not in visited_states and successor_state not in (state[0] for state in queue_state_path.list):

                    # Lectures code:
                    # All implementations run in autograder and in comments, I write
                    # the proper code that I have been taught in lectures
                    # if problem.isGoalState(successor_state):
                    #   return current_path + [action]

                    new_path = current_path + [action]  # Calculate new path
                    queue_state_path.push((successor_state, new_path))

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
