#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
	from PyQt6.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
from queue import PriorityQueue
import copy



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

# This function utilizes a greedy algorithm to find a solution to the traveling sales-person problem.
# It loops through the cities, and if the cost to a city in the list is less than the current
# distance, then the current distance becomes the small distance that is found. The city that
# has the minimum distance is also kept track of. The minimum city is then removed from the queue.
# If a minimum city was found, that becomes the current city. If there is a path from the current
# city back to the start city, it is appended to the path list. At the end, a BSSF is calculated of
# the path. The result dictionary is returned.
# Total Time Complexity: O(n^2)
# Total Space Complexity: O(n)
	def greedy( self,time_allowance=60.0 ):

		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		num_solutions = 0
		start_time = time.time()
		copy_cities = [cities[city] for city in range(ncities)]
		index_city = 0

		while not foundTour and time.time()-start_time < time_allowance:
			cities = [copy_cities[city] for city in range(len(copy_cities))]
			path = []
			city = cities.pop(index_city)
			index_city += 1
			start = city
			path.append(start)
			while time.time()-start_time < time_allowance and len(cities) > 0:

				minimum_city = None
				distance = float("inf")
				index = -1
				# loop through the cities
				for i in range(len(cities)):
					# checks if a cost to a city is less than the current distance
					if city.costTo(cities[i]) < distance:
						# updates the minimum distance and city
						distance = city.costTo(cities[i])
						minimum_city = cities[i]
						index = i
				cities.pop(index)

				if minimum_city != None:
					city = minimum_city
					if len(cities) == 0:
						# make sure there is a path from city, back to start city
						if minimum_city.costTo(start) < float("inf"):
							path.append(city)
							foundTour = True
							break
						else:
							foundTour = False
							num_solutions += 1
							break
					path.append(city)
				else:
					# no solution is found
					foundTour = False
					num_solutions += 1
					break

		bssf = TSPSolution(path)
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = num_solutions
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results



	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

# This function utilizes the Branch and Bound algorithm to solve the traveling sales-person problem
# It starts by finding a BSSF using the greedy algorithm. It creates a priority queue, and initializes
# the root state of the tree. It then puts the root state on the queue. Then, while the queue still
# has states in it, they are removed from the beginning of the queue. If the cost of the state pulled
# off the queue has a cost larger than the BSSF, it is pruned. I then loop through every city to make
# each child state. If the cost of the current state is not infinity, it means there is a path from this
# city to the next city. So, I initialize the child state, making the respective row and column of the
# state equal to infinity. If the state is not a leaf node, and it is less than the BSSF, update BSSF.
# The child state of the updates BSSF is added to the queue.
# Total Time Complexity: O((n^2)(2^n))
# Total Space Complexity: O((n^2)(2^n))
	def branchAndBound( self, time_allowance=60.0 ):

		results = {}
		self.cities = self._scenario.getCities()
		self.ncities = len(self.cities)
		num_states = 0
		num_solutions = 0
		max_queue = 0
		num_pruned = 0

		# finds an initial using greedy algorithm
		bssf = self.greedy(time_allowance)['soln']		# Time: O(n^2), Space: O(n)

		# initialize the queue
		queue = PriorityQueue()

		# creates root of tree
		root = self.State()
		root.costs = [[-1 for i in range(self.ncities)] for j in range(self.ncities)]
		self.state_initialization(None, root)			# Time: O(n^2)
		root.city = 0
		root.path.append(root.city)
		lower_bound = root.cost

		# adds the root to the queue
		queue.put((root.cost, root))

		start_time = time.time()

		# iterates 2^n times with pruning
		while not queue.empty() and time.time()-start_time < time_allowance:
			if queue.qsize() > max_queue:
				max_queue = queue.qsize()
			# grab the first state off the queue
			state = queue.get()[1]
			# if the cost of this state is greater than BSSF, prune it
			if state.cost > bssf.cost:
				num_pruned += 1
				continue
			# make each subproblem state
			for n in range(self.ncities):
				# there is a path from this city to next
				if state.costs[state.city][n] != float("inf"):
					# create a state for the subproblem
					child = self.State()
					self.state_initialization(state, child, n)  	# Time: O(n^2)
					num_states += 1

					#setting rows and columns to infinity: O(n)
					row = child.parent.city
					column = child.city
					# set respective row to infinity
					for j in range(self.ncities):
						child.costs[row][j] = float("inf")
					# set respective column to infinity
					for j in range(self.ncities):
						child.costs[j][column] = float("inf")

					# so you don't get premature cycles
					path_distance = len(child.path)
					index = path_distance - 1
					while index >= 0:
						row = child.city
						column = child.path[index]
						child.costs[row][column] = float("inf")
						index -= 1

					# reduce the state's cost matrix
					reduction = self.reduce(child)				# Time: O(n^2)
					step = child.parent.costs[child.parent.city][child.city]
					previous_state = child.parent.cost
					child.cost = previous_state + step + reduction

					# if state is a leaf node
					if len(child.path) == self.ncities:
						# if state is less than BSSF
						if child.cost < bssf.cost:
							path = []
							for k in range(self.ncities):
								path.append(self.cities[child.path[k]])
							# update BSSF
							if path[-1].costTo(path[0]) == float("inf"):
								continue
							else:
								bssf = TSPSolution(path)
								num_solutions += 1
						continue
					# add state to the queue
					if bssf.cost > child.cost > lower_bound:
						queue.put(((child.cost / child.depth), child))
					else:
						num_pruned += 1

		end_time = time.time()
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = num_solutions
		results['soln'] = bssf
		results['max'] = max_queue
		results['total'] = num_states
		results['pruned'] = num_pruned
		return results


# This function accepts a parent and child node, and initiates the state using the state using
# variables in the state class. It creates a cost matrix containing the costs from each city
# to the other cities.
# Total Time Complexity: O(n^2)
# Total Space Complexity: O(1)
	def state_initialization(self, parent, child, n=0):
		# You are at the root of the state tree
		if parent == None:
			root = child
			root.city = 0
			root.depth = 1
			for i in range(self.ncities):
				for j in range(self.ncities):
					root.costs[i][j] = self.cities[i].costTo(self.cities[j])
			root.cost = self.reduce(root)		# Time: O(n^2)
		# you are at a child state
		else:
			child.parent = parent
			child.city = n
			child.depth = child.parent.depth + 1
			child.costs = copy.deepcopy(child.parent.costs)
			child.path = copy.deepcopy(child.parent.path)
			child.path.append(n)

# This function reduces the cost matrix of a given state that is passed into the function.
# It loops through the matrix and begins by reducing the rows. It finds the minimum value
# in the row. If that value is not infinity or zero, the values in the row are updated so
# that cost can be subtracted from each value in the row. This same idea is repeated for
# column. As the matrix is reduced, I keep track of the cost. Once the function is complete,
# the function returns the cost.
# Total Time Complexity: O(n^2)
# Total Space Complexity: O(n^2)
	def reduce(self, state):
		cost = 0
		# reduces rows--> O(n^2)
		for i in range(self.ncities):			# O(n)
			min_row = float("inf")
			for j in range(self.ncities):		# O(n)
				if state.costs[i][j] < min_row:
					min_row = state.costs[i][j]
			if min_row != float("inf") and min_row != 0:
				cost += min_row
				for j in range(self.ncities):
					state.costs[i][j] -= min_row

		# reduces columns -> O(n^2)
		for j in range(self.ncities):			# O(n)
			min_col = float("inf")
			for i in range(self.ncities):		# O(n)
				if state.costs[i][j] < min_col:
					min_col = state.costs[i][j]
			if min_col != float("inf") and min_col != 0:
				cost += min_col
				for i in range(self.ncities):
					state.costs[i][j] -= min_col
		return cost

# This is a class to hold information on each state matrix to store the path so far,
# the parent node, the depth, the cost, and more.
	class State:
		# Time Complexity: O(1)
		# Space Complexity: O(1)
		def __init__(self):
			self.path = []
			self.costs = [[None], [None]]
			self.parent = None
			self.depth = 0
			self.cost = -1
			self.city = None

		def __gt__(self, other):
			if self.cost > other.cost:
				return True
			else:
				return False


	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

	def fancy( self,time_allowance=60.0 ):
		#divide and conquer ? ---> two different paths
		#Not all that glitters is gold
		pass
