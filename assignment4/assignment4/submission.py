from __future__ import print_function

import collections, util, copy
from itertools import product

############################################################
# Problem 3.1a

def create_nqueens_csp(n = 8):
    """
    Return an N-Queen problem on the board of size |n| * |n|.
    You should call csp.add_variable() and csp.add_binary_potential().

    @param n: number of queens, or the size of one dimension of the board.

    @return csp: A CSP problem with correctly configured potentials
        such that it can be solved by a weighted CSP solver.
    """
    csp = util.CSP()
    # BEGIN_YOUR_CODE (around 7 lines of code expected)
    # raise Exception("Not implemented yet")
    queens = range(n)
    for i in queens:
        domain = []
        for j in range(n):
            domain.append((i, j))
        csp.add_variable(i, domain)

    for q1 in queens:
        for q2 in queens:
            if q1 != q2:
                csp.add_binary_potential(q1, q2, lambda x, y : x[0] != y[0])
                csp.add_binary_potential(q1, q2, lambda x, y : x[1] != y[1])
                csp.add_binary_potential(q1, q2, lambda x, y : 
                                         abs(x[0] - y[0]) != abs(x[1] - y[1]))
    # END_YOUR_CODE
    return csp


# create_nqueens_csp()
############################################################
# Problem 3.1b

# A backtracking algorithm that solves weighted CSP.
# Usage:
#   search = BacktrackingSearch()
#   search.solve(csp)
class BacktrackingSearch():

    def reset_results(self):
        """
        This function resets the statistics of the different aspects of the
        CSP sovler. We will be using the values here for grading, so please
        do not make any modification to these variables.
        """
        # Keep track of the best assignment and weight found.
        self.optimalAssignment = {}
        self.optimalWeight = 0

        # Keep track of the number of optimal assignments and assignments. These
        # two values should be identical when the CSP is unweighted or only has binary
        # weights.
        self.numOptimalAssignments = 0
        self.numAssignments = 0

        # Keep track of the number of times backtrack() gets called.
        self.numOperations = 0

        # Keey track of the number of operations to get to the very first successful
        # assignment (doesn't have to be optimal).
        self.firstAssignmentNumOperations = 0

        # List of all solutions found.
        self.allAssignments = [] 
        self.allWeights = []

    def print_stats(self):
        """
        Prints a message summarizing the outcome of the solver.
        """
        if self.optimalAssignment:
            print ("Found %d optimal assignments with weight %f in %d operations" % \
                (self.numOptimalAssignments, self.optimalWeight, self.numOperations))
            print ("First assignment took %d operations" % self.firstAssignmentNumOperations)
        else:
            print ("No solution was found.")

    def get_delta_weight(self, assignment, var, val):
        """
        Given a CSP, a partial assignment, and a proposed new value for a variable,
        return the change of weights after assigning the variable with the proposed
        value.

        @param assignment: A list of current assignment. len(assignment) should
            equal to self.csp.numVars. Unassigned variables have None values, while an
            assigned variable has the index of the value with respect to its
            domain. e.g. if the domain of the first variable is [5,6], and 6
            was assigned to it, then assignment[0] == 1.
        @param var: Index of an unassigned variable.
        @param val: Index of the proposed value with resepct to |var|'s domain.

        @return w: Change in weights as a result of the proposed assignment. This
            will be used as a multiplier on the current weight.
        """
        assert assignment[var] is None
        w = 1.0
        if self.csp.unaryPotentials[var]:
            w *= self.csp.unaryPotentials[var][val]
            if w == 0: return w
        for var2, potential in self.csp.binaryPotentials[var].items():
            if assignment[var2] == None: continue  # Not assigned yet
            w *= potential[val][assignment[var2]]
            if w == 0: return w
        return w

    def solve(self, csp, mcv = False, lcv = False, mac = False):
        """
        Solves the given weighted CSP using heuristics as specified in the
        parameter. Note that unlike a typical unweighted CSP where the search
        terminates when one solution is found, we want this function to find
        all possible assignments. The results are stored in the variables
        described in reset_result().

        @param csp: A weighted CSP.
        @param mcv: When enabled, Monst Constrained Variable heuristics is used.
        @param lcv: When enabled, Least Constraining Value heuristics is used.
        @param mac: When enabled, AC-3 will be used after each assignment of an
            variable is made.
        """
        # CSP to be solved.
        self.csp = csp

        # Set the search heuristics requested asked.
        self.mcv = mcv
        self.lcv = lcv
        self.mac = mac

        # Reset solutions from previous search.
        self.reset_results()

        # The list of domains of every variable in the CSP. Note that we only
        # use the indices of the values. That is, if the domain of a variable
        # A is [2,3,5], then here, it will be stored as [0,1,2]. Original domain
        # name/value can be obtained from self.csp.valNames[A]
        self.domains = [list(range(len(domain))) for domain in self.csp.valNames]
        # Perform backtracking search.
        self.backtrack([None] * self.csp.numVars, 0, 1)

        # Print summary of solutions.
        self.print_stats()

    def backtrack(self, assignment, numAssigned, weight):
        """
        Perform the back-tracking algorithms to find all possible solutions to
        the CSP.

        @param assignment: A list of current assignment. len(assignment) should
            equal to self.csp.numVars. Unassigned variables have None values, while an
            assigned variable has the index of the value with respect to its
            domain. e.g. if the domain of the first variable is [5,6], and 6
            was assigned to it, then assignment[0] == 1.
        @param numAssigned: Number of currently assigned variables
        @param weight: The weight of the current partial assignment.
        """

        self.numOperations += 1
        assert weight > 0
        if numAssigned == self.csp.numVars:
            # A satisfiable solution have been found. Update the statistics.
            self.numAssignments += 1
            newAssignment = {}
            for var in range(self.csp.numVars):
                newAssignment[self.csp.varNames[var]] = self.csp.valNames[var][assignment[var]]
            self.allAssignments.append(newAssignment)
            self.allWeights.append(weight)

            if len(self.optimalAssignment) == 0 or weight >= self.optimalWeight:
                if weight == self.optimalWeight:
                    self.numOptimalAssignments += 1
                else:
                    self.numOptimalAssignments = 1
                self.optimalWeight = weight

                self.optimalAssignment = newAssignment
                if self.firstAssignmentNumOperations == 0:
                    self.firstAssignmentNumOperations = self.numOperations
            return

        # Select the index of the next variable to be assigned.
        var = self.get_unassigned_variable(assignment)

        # Obtain the order of which a variable's values will be tried. Note that
        # this stores the indices of the values with respect to |var|'s domain.
        ordered_values = self.get_ordered_values(assignment, var)
        # Continue the backtracking recursion using |var| and |ordered_values|.
        if not self.mac:
            # When arc consistency check is not enabled.
            for val in ordered_values:
                deltaWeight = self.get_delta_weight(assignment, var, val)
                if deltaWeight > 0:
                    assignment[var] = val
                    self.backtrack(assignment, numAssigned + 1, weight * deltaWeight)
                    assignment[var] = None
        else:
            # Problem 3.1d
            # When arc consistency check is enabled.
            # BEGIN_YOUR_CODE (around 10-15 lines of code expected)
            # raise Exception("Not implemented yet")
            cur_domain = copy.deepcopy(self.domains)
            for val in ordered_values:
                self.domains = copy.deepcopy(cur_domain)
                assignment[var] = val
                self.domains[var] = [val]
                self.arc_consistency_check(var)
                self.backtrack(assignment, numAssigned + 1, weight)
                assignment[var] = None
            # END_YOUR_CODE

    def get_unassigned_variable(self, assignment):
        """
        Given a partial assignment, return the index of a currently unassigned
        variable.

        @param assignment: A list of current assignment. This is the same as
            what you've seen so far.

        @return var: Index of a currently unassigned variable.
        """

        if not self.mcv:
            # Select a variable without any heuristics.
            for var in range(len(assignment)):
                if assignment[var] is None: return var
        else:
            # Problem 3.1b
            # Heuristic: most constrained variable (MCV)
            # Select a variable with the least number of remaining domain values.
            # BEGIN_YOUR_CODE (around 5 lines of code expected)
            # raise Exception("Not implemented yet")
            numAssign = len(assignment)
            varList = []
            for var in xrange(numAssign):
                if assignment[var] is None:
                    domain = self.domains[var]
                    aCount = 0
                    for val in domain:
                        delta = self.get_delta_weight(assignment, var, val)
                        if delta > 0:
                            aCount += 1
                    varList.append( (aCount, var) )
            return min(varList)[1]
            # END_YOUR_CODE

    def get_ordered_values(self, assignment, var):
        """
        Given an unassigned variable and a partial assignment, return an ordered
        list of indices of the variable's domain such that the backtracking
        algorithm will try |var|'s values according to this order.

        @param assignment: A list of current assignment. This is the same as
            what you've seen so far.
        @param var: The variable that's going to be assigned next.

        @return ordered_values: A list of indeces of |var|'s domain values.
        """
        if not self.lcv:
            # Return an order of value indices without any heuristics.
            # if var == None:
            #     import pdb; pdb.set_trace()
            return self.domains[var]
        else:
            # Problem 3.1c
            # Heuristic: least constraining value (LCV)
            # Return value indices in ascending order of the number of additional
            # constraints imposed on unassigned neighboring variables.
            # BEGIN_YOUR_CODE (around 12 lines of code expected)
            # Will update the domains! The unary constraint on var, val was already checked by backtrack before calling this method
            # raise Exception("Not implemented yet")
            values = self.domains[var]
            neighbors = self.csp.binaryPotentials[var]

            neighbor_values = []

            for n in neighbors:
                vals = []
                for i in values:
                    count = 0
                    for j in range(len(neighbors[n][i])):
                        if neighbors[n][i][j] != 0.0:
                            count += 1
                    vals.append((i, count))
                neighbor_values.append(vals)

            max_sum = None
            max_list = None
            for nv in neighbor_values:
                curSum = sum([n[1] for n in nv])
                if max_sum == None or curSum > max_sum:
                    max_sum = curSum
                    max_list = nv

            max_list.sort(key=lambda tup: tup[1])
            return [elem[0] for elem in max_list]
            # END_YOUR_CODE
   
    def arc_consistency_check(self, var):
        """
        Perform the AC-3 algorithm. The goal is to reduce the size of the
        domain values for the unassigned variables based on arc consistency.

        @param var: The variable whose value has just been set.

        While not required, you can also choose to add return values in this
        function if there's a need.
        '''
            csp.binaryPotentials = 
            [{1: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              5: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]},

             {0: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              2: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              5: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]},

             {1: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              3: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              5: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]},

             {2: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              4: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              5: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]},

             {3: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              5: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]},

             {0: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              1: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              2: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              3: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              4: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]},

             {}]
        '''
        """
        # Problem 3.1d
        # BEGIN_YOUR_CODE (around 15-20 lines of code expected)
        # raise Exception("Not implemented yet")
        queue = [var]
        while len(queue) != 0:
            x = queue[0]
            queue.remove(x)
            neighbors = self.csp.binaryPotentials[x]
            vals = self.domains[x]
            for n in neighbors:
                nvals = copy.deepcopy(self.domains[n])
                domain = set()
                for i in vals:
                    for j in nvals:
                        if neighbors[n][i][j] != 0.0:
                            domain.add(j)
                if len(domain) != len(nvals):
                    queue.append(n)
                self.domains[n] = list(domain)
        # END_YOUR_CODE

############################################################
# Problem 3.2

def get_sum_variable(csp, name, variables, maxSum):
    """
    Given a list of |variables| each with non-negative integer domains,
    returns the name of a new variable with domain [0, maxSum], such that
    it's consistent with the value |n| iff the assignments for |variables|
    sums to |n|.

    @param name: Prefix of all the variables that are going to be added.
        Can be any hashable objects. For every variable |var| added in this
        function, it's recommended to use a naming strategy such as
        ('sum', |name|, |var|) to avoid conflicts with other variable names.
    @param variables: A list of variables that are already in the CSP that
        have non-negative integer values as its domain.
    @param maxSum: An integer indicating the maximum sum value allowed.

    @return result: The name of a newly created variable with domain
        [0, maxSum] such that it's consistent with an assignment of |n|
        iff the assignment of |variables| sums to |n|.
    """

    # BEGIN_YOUR_CODE (around 12-15 lines of code expected)
    # raise Exception("Not implemented yet")
    prefix = "sum" + name
    if len(variables) == 0:
        final_prefix = prefix + "final"
        csp.add_variable(final_prefix, [0])
        return final_prefix
    for i in range(len(variables)):
        aux = prefix + str(i) + "aux"
        csp.add_variable(aux, [(x,y) for x,y in product(range(maxSum + 1), range(maxSum + 1))])
        csp.add_binary_potential(variables[i], aux, lambda x,y: y[1] == y[0] + x)
        if i > 0:
            csp.add_binary_potential(aux, prefix + str(i-1) + "aux", lambda x,y: x[0]== y[1])
        else:
            csp.add_unary_potential(aux, lambda x: x[0] == 0)
    last_prefix = prefix + str(len(variables) - 1) + "aux"
    final_prefix = prefix + "final"
    csp.add_variable(final_prefix, range(maxSum + 1))
    csp.add_binary_potential(last_prefix, final_prefix, lambda x, y: y == x[1])
    return final_prefix
    # END_YOUR_CODE

def get_or_variable(csp, name, variables, value):
    """
    Create a new variable with domain [True, False] that can only be assigned to
    True iff at least one of the |variables| is assigned to |value|. You should
    add any necessary intermediate variables, unary potentials, and binary
    potentials to achieve this. Then, return the name of this variable.

    @param name: Prefix of all the variables that are going to be added.
        Can be any hashable objects. For every variable |var| added in this
        function, it's recommended to use a naming strategy such as
        ('or', |name|, |var|) to avoid conflicts with other variable names.
    @param variables: A list of variables in the CSP that are participating
        in this OR function. Note that if this list is empty, then the returned
        variable created should never be assigned to True.
    @param value: For the returned OR variable being created to be assigned to
        True, at least one of these variables must have this value.

    @return result: The OR variable's name. This variable should have domain
        [True, False] and constraints s.t. it's assigned to True iff at least
        one of the |variables| is assigned to |value|.
    """

    # BEGIN_YOUR_CODE (around 20 lines of code expected)
    prefix = "or" + name
    if len(variables) == 0:
        final_prefix = prefix + "final"
        csp.add_variable(final_prefix, [False])
        return final_prefix
    for i in range(len(variables)):
        aux = prefix + str(i) + "aux"
        csp.add_variable(aux, [(True, True), (True, False), (False, True), (False, False)])
        csp.add_binary_potential(variables[i], aux, lambda x, y: y[1] == ((x == value) or y[0]))
        if i > 0:
            csp.add_binary_potential(aux, prefix + str(i-1) + "aux", lambda x, y: x[0] == y[1])
        else:
            csp.add_unary_potential(aux, lambda x: not x[0])
    last_prefix = prefix + str(len(variables) - 1) + "aux"
    final_prefix = prefix + "final"
    csp.add_variable(final_prefix, [True, False])
    csp.add_binary_potential(last_prefix, final_prefix, lambda x, y: y == x[1])
    return final_prefix

############################################################
# Problem 3.3

# A class providing methods to generate CSP that can solve the course scheduling
# problem.
class SchedulingCSPConstructor():

    def __init__(self, bulletin, profile):
        """
        Saves the necessary data.

        @param bulletin: Rice Bulletin that provides a list of courses
        @param profile: A student's profile and requests
        """
        self.bulletin = bulletin
        self.profile = profile

    def add_variables(self, csp):
        """
        Adding the variables into the CSP. Each variable, (req, course),
        can take on the value of one of the semesters in req or None.
        For instance, for course='COMP310', and a request object, req, generated
        from 'in Fall2013,Fall2014', then (req, course) should have the domain values
        ['Fall2013', 'Fall2014', None]. Conceptually, if var is assigned 'Fall2013'
        then it means we are taking 'COMP310' in 'Fall2013'. If it's None, then
        we not taking COMP310.

        @param csp: The CSP where the additional constraints will be added to.
        """

        for req in self.profile.requests:
            for cid in req.cids:
                if cid not in csp.varNames:
                    csp.add_variable(cid, self.profile.semesters + [None])
                for cid2 in req.prereqs:
                    if cid2 not in self.profile.taken:
                        if cid2 not in csp.varNames:
                            csp.add_variable(cid2, self.profile.semesters + [None])
                #the values could be one of the semesters or None

    def add_bulletin_constraints(self, csp):
        """
        Add the constraints that a course can only be taken if it's offered in
        that semester.

        @param csp: The CSP where the additional constraints will be added to.
        """
        for cid in csp.varNames:
            csp.add_unary_potential(cid, \
                lambda semester: semester is None or self.bulletin.courses[cid].is_offered_in(semester) )

    def get_basic_csp(self):
        """
        Return a CSP that only enforces the basic constraints that a course can
        only be taken when it's offered and that a request can only be satisfied
        in at most one semester.

        @return csp: A CSP where basic variables and constraints are added.
        """
        csp = util.CSP()
        self.add_variables(csp)
        self.add_bulletin_constraints(csp)
        return csp

    def add_semester_constraints(self, csp):
        """
        If the profile explicitly wants a request to be satisfied in some given
        semesters, e.g. Fall2013, then add constraints to not allow that request to be satisified in any other semester.

        @param csp: The CSP where the additional constraints will be added to.
        """
        # BEGIN_YOUR_CODE (around 4 lines of code expected)      
        # raise Exception("Not implemented yet")
        for req in self.profile.requests: 
            for cid in req.cids:
                for semester in self.profile.semesters:
                    csp.add_unary_potential(cid, lambda semester: semester is None or semester in req.semesters)
        # END_YOUR_CODE

    def add_request_weights(self, csp):
        """
        Incorporate weights into the CSP. By default, a request has a weight
        value of 1 (already configured in Request). You should only use the
        weight when one of the requested course is in the solution. A
        unsatisfied request should also have a weight value of 1.

        @param csp: The CSP where the additional constraints will be added to.
        """
        # BEGIN_YOUR_CODE (around 3 lines of code expected)      
        # raise Exception("Not implemented yet")
        for req in self.profile.requests:
            for cid in req.cids: 
                csp.add_unary_potential(cid, lambda x: req.weight if x != None else 1)
        
        # END_YOUR_CODE

    def add_prereq_constraints(self, csp):
        """
        Adding constraints to enforce prerequisite. A course can have multiple
        prerequisites. You can assume that all courses in req.prereqs are
        being requested. Note that req.prereqs apply to every single course in 
        req.cids. You cannot take a course in a semester unless all of its 
        prerequisites have been taken before that semester. You should write your 
        own function that check the values (i.e. semesters) of the course you 
        request and its prerequisites and make sure that the values of prerequisites 
        are smaller (e.g. Spr2014 is smaller than Fall2015) than that of the course 
        you request if not None.

        @param csp: The CSP where the additional constraints will be added to.
        """
        # BEGIN_YOUR_CODE (around 20 lines of code expected)      
        # raise Exception("Not implemented yet")
        def temp(x, y):
            if x is None or y is None:
                return False
            semX, yearX = x[:-4], x[-4:]
            semY, yearY = y[:-4], y[-4:]
            semX = 0 if semX == 'Fall' else 1
            semY = 0 if semY == 'Fall' else 1
            if yearX == yearY:
            	return semX > semY
            return yearX > yearY

        for req in self.profile.requests:
            if len(req.prereqs) == 0: 
                continue 
            for cid in req.cids:
                for preq in req.prereqs:
                    # print(csp.valNames[cid], csp.valNames[preq])    
                    csp.add_binary_potential(cid, preq, lambda x, y: x == None or temp(x, y) )

        # END_YOUR_CODE

    def add_credit_constraints(self, csp):
        """
        Add constraint to the CSP to ensure that the total number of units are
        within profile.minUnits/maxmaxUnits, inclusively. The allowed range for
        each course can be obtained from bulletin.courses[id].minUnits/maxUnits.
        For a request 'Request A', if you choose to take A, then you must use a unit
        number that's within the range of A. You should introduce any additional
        variables that you are needed. In order for our solution extractor to
        obtain the number of units, for every course that you plan to take in
        the solution, you must have a variable named (courseId, semester) (e.g.
        ('COMP182', 'Fall2013') and it's assigned value is the number of units.
        You should take advantage of get_sum_variable().

        Note 1:
        In here, you can not use 
        for req in requests: 
            for course in req:
        to loop over all the courses, because prereqs are not added in to the instances 
        of request object.

        Note 2:
        So you will have to loop over variables in the csp. But there
        are different types of variables: courses and auxiliary variables with
        which you handle sums (e.g. (courseId, semester) and those added in get_sum_variable). 
        Please check the types of the variables before you work with them.
        

        @param csp: The CSP where the additional constraints will be added to.
        """
        # BEGIN_YOUR_CODE (around 13-15 lines of code expected)
        # raise Exception("Not implemented yet")
        for quarter in self.profile.semesters:
            quarter_variables = []
            for req in self.profile.requests:
                for cid in req.cids:
                    csp.add_variable((cid, quarter), 
                                     [0] + range(self.bulletin.courses[cid].minUnits, self.bulletin.courses[cid].maxUnits + 1))
                    quarter_variables.append((cid, quarter))
                    csp.add_binary_potential((req, quarter), (cid, quarter), 
                                             lambda x, y: y > 0 if x == cid else y == 0)
            quarter_max_sum = get_sum_variable(csp, "quarter" + quarter, 
                                               quarter_variables, 
                                               self.profile.maxUnits)
            csp.add_unary_potential(quarter_max_sum, lambda x: x >= self.profile.minUnits and x <= self.profile.minUnits)
        # END_YOUR_CODE

    def add_all_additional_constraints(self, csp):
        """
        Add all additional constraints to the CSP.

        @param csp: The CSP where the additional constraints will be added to.
        """
        self.add_semester_constraints(csp)
        self.add_request_weights(csp)
        self.add_prereq_constraints(csp)
        self.add_credit_constraints(csp)
