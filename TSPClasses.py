#!/usr/bin/python3


import math
import numpy as np
import random
import time



class TSPSolution:
	def __init__( self, listOfCities):
		self.route = listOfCities
		self.cost = self._costOfRoute()
		#print( [c._index for c in listOfCities] )

	def _costOfRoute( self ):
		cost = 0
		last = self.route[0]
		for city in self.route[1:]:
			cost += last.costTo(city)
			last = city
		cost += self.route[-1].costTo( self.route[0] )
		return cost

	def enumerateEdges( self ):
		elist = []
		c1 = self.route[0]
		for c2 in self.route[1:]:
			dist = c1.costTo( c2 )
			if dist == np.inf:
				return None
			elist.append( (c1, c2, int(math.ceil(dist))) )
			c1 = c2
		dist = self.route[-1].costTo( self.route[0] )
		if dist == np.inf:
			return None
		elist.append( (self.route[-1], self.route[0], int(math.ceil(dist))) )
		return elist


def nameForInt( num ):
	if num == 0:
		return ''
	elif num <= 26:
		return chr( ord('A')+num-1 )
	else:
		return nameForInt((num-1) // 26 ) + nameForInt((num-1)%26+1)








class Scenario:

	HARD_MODE_FRACTION_TO_REMOVE = 0.20 # Remove 20% of the edges

	def __init__( self, city_locations, difficulty, rand_seed ):
		self._difficulty = difficulty

		if difficulty == "Normal" or difficulty == "Hard":
			self._cities = [City( pt.x(), pt.y(), \
								  random.uniform(0.0,1.0) \
								) for pt in city_locations]
		elif difficulty == "Hard (Deterministic)":
			random.seed( rand_seed )
			self._cities = [City( pt.x(), pt.y(), \
								  random.uniform(0.0,1.0) \
								) for pt in city_locations]
		else:
			self._cities = [City( pt.x(), pt.y() ) for pt in city_locations]


		num = 0
		for city in self._cities:
			#if difficulty == "Hard":
			city.setScenario(self)
			city.setIndexAndName( num, nameForInt( num+1 ) )
			num += 1

		# Assume all edges exists except self-edges
		ncities = len(self._cities)
		self._edge_exists = ( np.ones((ncities,ncities)) - np.diag( np.ones((ncities)) ) ) > 0

		if difficulty == "Hard":
			self.thinEdges()
		elif difficulty == "Hard (Deterministic)":
			self.thinEdges(deterministic=True)

	def getCities( self ):
		return self._cities


	def randperm( self, n ):				#isn't there a numpy function that does this and even gets called in Solver?
		perm = np.arange(n)
		for i in range(n):
			randind = random.randint(i,n-1)
			save = perm[i]
			perm[i] = perm[randind]
			perm[randind] = save
		return perm

	def thinEdges( self, deterministic=False ):
		ncities = len(self._cities)
		edge_count = ncities*(ncities-1) # can't have self-edge
		num_to_remove = np.floor(self.HARD_MODE_FRACTION_TO_REMOVE*edge_count)

		can_delete	= self._edge_exists.copy()

		# Set aside a route to ensure at least one tour exists
		route_keep = np.random.permutation( ncities )
		if deterministic:
			route_keep = self.randperm( ncities )
		for i in range(ncities):
			can_delete[route_keep[i],route_keep[(i+1)%ncities]] = False

		# Now remove edges until 
		while num_to_remove > 0:
			if deterministic:
				src = random.randint(0,ncities-1)
				dst = random.randint(0,ncities-1)
			else:
				src = np.random.randint(ncities)
				dst = np.random.randint(ncities)
			if self._edge_exists[src,dst] and can_delete[src,dst]:
				self._edge_exists[src,dst] = False
				num_to_remove -= 1




class City:
	def __init__( self, x, y, elevation=0.0 ):
		self._x = x
		self._y = y
		self._elevation = elevation
		self._scenario	= None
		self._index = -1
		self._name	= None

	def setIndexAndName( self, index, name ):
		self._index = index
		self._name = name

	def setScenario( self, scenario ):
		self._scenario = scenario

	''' <summary>
		How much does it cost to get from this city to the destination?
		Note that this is an asymmetric cost function.
		 
		In advanced mode, it returns infinity when there is no connection.
		</summary> '''
	MAP_SCALE = 1000.0
	def costTo( self, other_city ):

		assert( type(other_city) == City )

		# In hard mode, remove edges; this slows down the calculation...
		# Use this in all difficulties, it ensures INF for self-edge
		if not self._scenario._edge_exists[self._index, other_city._index]:
			return np.inf

		# Euclidean Distance
		cost = math.sqrt( (other_city._x - self._x)**2 +
						  (other_city._y - self._y)**2 )

		# For Medium and Hard modes, add in an asymmetric cost (in easy mode it is zero).
		if not self._scenario._difficulty == 'Easy':
			cost += (other_city._elevation - self._elevation)
			if cost < 0.0:
				cost = 0.0					# Shouldn't it cost something to go downhill, no matter how steep??????


		return int(math.ceil(cost * self.MAP_SCALE))

#cluster of cities is called neighborhood
class Neighborhood:
	"""
	__ methods should inlcude init and average distance
	other method needed are shortest path of neighborhoods, path between neightborhoods, merge
	"""
	def __init__(self, route: list):
		#set to 0
		self.route = route
		self.avg_x = 0.
		self.avg_y = 0.
		self.avg_elev = 0.

		#to optimize merge average f city
		for city in route:
			self.avg_x += city._x
			self.avg_y += city._y
			self.avg_elev += city._elevation
		#now do the divide part of average
		self.avg_x /= len(route)
		self.avg_y /= len(route)
		self.avg_elev /= len(route)

	#gets distance average between city1 and city2
	def __average_distance(self, city1, city2):
		#used city as example
		# Euclidean Distance
		cost = math.sqrt( (city2.avg_x - city1._x)**2 + 
						  (city2.avg_y - city1._y)**2 )
		#need to account for elevation as part of distance (z)
		cost += (city2.avg_elev - city1._elevation)
		#reset cost to 0 if less than 0 needs to be more exact (not limited to strictly int)
		if cost < 0.0: cost = 0.0
		return int(math.ceil(cost * 1000.0))

	#I think this is nm where n is len of neighborhood and m is len of other_cities
	def shortest_neightborhood_path(self, other_city):
		#city 1 to city 2 and city 3 to city 4
		#make neighborhood to be sorted by distance
		neighborhood = sorted(self.route.copy(), key=lambda c: self.__average_distance(c, other_city))
		for city_1 in neighborhood:
			#make var other cities to be sorfted by cost
			other_cities = sorted(other_city.route.copy(), key=lambda c: city_1.costTo(c))
			for city_2 in other_cities:
				#check route of city 2 modded by len of route
				city_3 = other_city.route[(other_city.route.index(city_2) - 1) % len(other_city.route)]
				#check route of city 1 modded by len of route
				city_4 = self.route[(self.route.index(city_1) + 1) % len(self.route)]
				#if cost of city 1 to 2 adn city 3 to 4 are not infinte return that path
				if city_1.costTo(city_2) < np.inf and city_3.costTo(city_4) < np.inf: return [city_1, city_2]
		return None
	
	#I think that is is nm where n is len of self.route and m is len of other_neighborhood.route
	def path_between_neighborhoods(self, other_neighborhood):
		#want to get teh min
		min_cost = np.inf
		min_city_1 = None
		min_city_2 = None
		#check for all city 1
		#similar to shortest nieghborhood path system byt instead of doing it by neighborhood do it by route
		for city_1 in self.route:
			for city_2 in other_neighborhood.route:
				city_3 = other_neighborhood.route[(other_neighborhood.route.index(city_2) - 1) % len(other_neighborhood.route)]
				city_4 = self.route[(self.route.index(city_1) + 1) % len(self.route)]
				cost = city_1.costTo(city_2) + city_3.costTo(city_4) - city_1.costTo(city_4) - city_3.costTo(city_2)
				#check cost 
				if cost < min_cost:
					min_cost = cost
					min_city_1 = city_1
					min_city_2 = city_2
		#chekc min costs retrun min city 1 & 2			
		if min_cost < np.inf: return [min_city_1, min_city_2]
		else: return None

	#I think this is nm where n is len of self.route and m is len of temp_route which is set to other_node.rout
	def merge_together(self, other_node):
		path_between = self.path_between_neighborhoods(other_node)
		#if no path is returned no path is possible... exit
		if path_between is None: return None
		new_route = []
		for city in self.route:
			#append city to the new orute
			new_route.append(city)
			#check if city is in path betweenk
			if city is path_between[0]:
				#need temp route and index for walking through honestly done for organization
				temp_route = other_node.route
				temp_index = temp_route.index(path_between[1])
				for i in range(len(temp_route)): new_route.append(temp_route[(temp_index + i) % len(temp_route)])
		return Neighborhood(new_route)
		