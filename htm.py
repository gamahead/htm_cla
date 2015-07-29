import random
import random as r
from math import *
import numpy

# =============================================
# Utilities
# =============================================
 
def partition(vector, left, right, pivotIndex):
    pivotValue = vector[pivotIndex]
    vector[pivotIndex], vector[right] = vector[right], vector[pivotIndex]  # Move pivot to end
    storeIndex = left
    for i in range(left, right):
        if vector[i] < pivotValue:
            vector[storeIndex], vector[i] = vector[i], vector[storeIndex]
            storeIndex += 1
    vector[right], vector[storeIndex] = vector[storeIndex], vector[right]  # Move pivot to its final place
    return storeIndex
 
def _select(vector, left, right, k):
    "Returns the k-th smallest, (k >= 0), element of vector within vector[left:right+1] inclusive."
    while True:
        pivotIndex = random.randint(left, right)     # select pivotIndex between left and right
        pivotNewIndex = partition(vector, left, right, pivotIndex)
        pivotDist = pivotNewIndex - left
        if pivotDist == k:
            return vector[pivotNewIndex]
        elif k < pivotDist:
            right = pivotNewIndex - 1
        else:
            k -= pivotDist + 1
            left = pivotNewIndex + 1
 
def select(vector, k, left=None, right=None):
    """
    Returns the k-th smallest, (k >= 0), element of vector within vector[left:right+1].
    left, right default to (0, len(vector) - 1) if omitted
    """
    if left is None:
        left = 0
    lv1 = len(vector) - 1
    if right is None:
        right = lv1
    assert vector and k >= 0, "Either null vector or k < 0 "
    assert 0 <= left <= lv1, "left is out of range"
    assert left <= right <= lv1, "right is out of range"
    return _select(vector, left, right, k)

# =============================================
# HTM Params & Basic Objects
# =============================================

class Synapse:
	def __init__(self, source_input, permanence = 0.5):
		self.source_input = source_input
		self.permanence = permanence

class Cell:
	def __init__(self):
		self.active_state = [r.randint(0,1), r.randint(0,1)]
		self.predictive_state = [r.randint(0,1), r.randint(0,1)]
		self.segments = []

	def get_active_segment(self, t, active_state):
		return([s for s in self.segments if s.state == 1])

class Segment:
	def __init__(self, source_cell):
		self.source_cell = source_cell
		self.state = r.randint(0,1)
		self.strength = r.uniform(0,1)

	def set_state(self):
		self.state = 1 if self.source_cell.active_state[-1] == 1 else 0

	def update(self):
		self.strength += update_function(self.source_cell.active_state[-1], self.strength)

	# Simple linear updating for now, but hope to implement some kind of logit function 
	# Modularized like this because I think the synapse updating function will be important
	def update_function(self, source_active, strength):
		return(max(min(strength + (.1 if source_active else -0.1), 1), 0))
		

class Column:
	def __init__(self, connected_synapses, cells_per_column = 5, x=0, y=0, hist_num=1000):
		self.connected_synapses = connected_synapses
		self.x = x
		self.y = y
		self.hist_num = hist_num # How much history do we want to keep?
		self.overlap = 0
		self.min_duty_cycle = 0
		self.active_duty_cycle_history = [r.randint(0,1) for _ in range(0,hist_num)] # Record last hist_num iterations
		self.overlap_duty_cycle_history = [r.randint(0,1) for _ in range(0,hist_num)] # And init with randoms
		self.active_duty_cycle = 0
		self.overlap_duty_cycle = 0
		self.boost = 1

		# TP
		self.cells = [Cell()] * cells_per_column
		self.predicted = 0
		self.predicted_w_strength = 0

class Retina:
	def __init__(self, x, y):
		self.x = x
		self.y = y

		self.receptors = numpy.zeros((x,y), dtype=numpy.int)

class HTM:
	def __init__(self, 
			num_columns = 25, 
			cells_per_column = 5,
			t = 1,
			inhibition_radius = 10,
			desired_local_activity = 10,
			connected_perm = .5,
			num_synapses = 10,
			min_overlap = 5):

		self.num_columns = num_columns
		self.cells_per_column = cells_per_column
		self.t = t
		self.inhibition_radius = inhibition_radius
		self.desired_local_activity = desired_local_activity
		self.connected_perm = connected_perm
		self.min_overlap = min_overlap
		self.columns = []

		self.x = 5
		self.y = 5
		i = 0
		j = 0
		for _ in range(0, self.num_columns):
			new_synapses = [(r.randint(0,self.x-1),r.randint(0,self.y-1)) for _ in range(0,num_synapses)]
			self.columns.append( Column( [Synapse(s) for s in new_synapses], x=i, y=j) )
			i += 1
			if i % 5 == 0:
				i = 0
				j += 1

		self.active_columns = [self.columns]
		self.input_space = Retina(self.x,self.y)

	def get_input(self, t, source_input):
		print(source_input)
		return(self.input_space.receptors[source_input])

	def set_input_space(self, new_inputs):
		self.input_space.receptors = new_inputs


# =============================================
# Spatial Pooling
# =============================================

def kth_score(columns, desired_local_activity):
	return select([c.overlap for c in columns], desired_local_activity)

def max_duty_cycle(columns):
	return max([c.active_duty_cycle for c in columns])	

# This implementation assumes 2D data because it was built for saccadic vision, 
# so the neighbors are considered the columns in a circular radius 
def neighbors(column, htm):
	x = column.x
	y = column.y
	return([c for c in htm.columns if sqrt((c.x - x)**2 + (c.y - y)**2) <= htm.inhibition_radius])

def boost_function(active_duty_cycle, min_duty_cycle):
	return(1 if active_duty_cycle >= min_duty_cycle else min_duty_cycle/active_duty_cycle)

def update_active_duty_cycle(column):
	column.active_duty_cycle_history.pop(0)
	column.active_duty_cycle_history.append(1 if overlap > 0 else 0)
	return(sum(column.active_duty_cycle_history)/column.hist_num)

def update_overlap_duty_cycle(column):
	column.overlap_duty_cycle.pop(0)
	column.overlap_duty_cycle_history.append(1 if overlap > 0 else 0)
	return(sum(column.overlap_duty_cycle_history)/column.hist_num)

def increase_permanences(column, perm_inc):
	for s in column.connected_synapses:
		s.permanence += perm_inc

def average_receptive_field_size(columns):

	def receptive_field_size(c):
		sum([1 for s in c.connected_synapses if s >= connected_perm])

	# return( sum( [receptive_field_size(c) for c in columns] ) / len(columns) )
	return(numpy.mean([receptive_field_size(c) for c in columns]))

# def get_input(t, source_input):
# 	return(1 if source_input else 0)
		

	
def Spatial(htm):

	# House keeping to fix indexing errors
	htm.active_columns.append([])

	###############################################
	# Phase 1: Overlap
	###############################################
	for c in htm.columns:
		c.overlap = 0

		for s in c.connected_synapses:
			c.overlap += htm.get_input(htm.t, s.source_input)

		if c.overlap < htm.min_overlap:
			c.overlap = 0
		else:
			c.overlap += c.boost

	###############################################
	# Phase 2: Inhibition
	###############################################
	for c in htm.columns:
		htm.min_local_activity = kth_score(neighbors(c, htm), htm.desired_local_activity)

		if c.overlap > 0 and c.overlap >= htm.min_local_activity:
			htm.active_columns[htm.t].append(c)

	###############################################
	# Phase 3: Learning
	###############################################
	for c in htm.active_columns[htm.t]:
		
		for s in c.potential_synapses:
			if s.active:
				s.permanence += htm.permanence_inc
				s.permanence = min(1.0, s.permanence)
			else:
				s.permanence -= htm.permanence_dec
				s.permanence = max(0.0, s.permanence)

	for c in htm.columns:
		c.min_duty_cycle = 0.01 * max_duty_cycle(neighbors(c),htm)
		c.active_duty_cycle = update_active_duty_cycle(c)
		c.boost = boost_function(c.active_duty_cycle, c.min_duty_cycle)

		c.overlap_duty_cycle = update_overlap_duty_cycle(c)
		if c.overlap_duty_cycle < c.min_duty_cycle:
			increase_permanences(c, 0.1*htm.connected_perm)

	htm.inhibition_radius = average_receptive_field_size(htm.columns)

# =============================================
# Temporal Pooling
# =============================================

def Temporal(htm):

	# House-keeping: add a 0 to the lists to keep index in-bounds
	for c in htm.columns:
		for cell in c.cells:
			cell.active_state.append(0)
			cell.predictive_state.append(0)


	###############################################
	# Phase 1: Inference and Learning
	###############################################

	for c in htm.active_columns[t]:

		bu_predicted = False
		lc_chosen = False
		for cell in c.cells:
			if cell.predictive_state[htm.t-1]:
				bu_predicted = True
				cell.active_state[htm.t] = 1

		if not bu_predicted:
			for cell in c.cells:
				cell.active_state[htm.t] = 1

		# Update segment strengths
		# Every segment gets strengthened if correct or weakened if incorrect prediction
		for cell in c.cells:
			for s in cell.segments:
				s.update()

	###############################################
	# Phase 2: Calculate Predictive States
	###############################################

	for c in htm.active_columns[htm.t]:
		c.predicted = 0
		for cell in c.cells:
			for s in cell.segments:
				if s.source_cell.active_state[htm.t]:
					c.predicted += 1
					cell.predictive_state[t] = 1

	htm.t += 1


if __name__ == '__main__':
    my_htm = HTM()
    input1 = numpy.array([1]*10 + [0]*15).reshape((5,5))
    input2 = numpy.array([0]*15 + [1]*10).reshape((5,5))
    for i in range(0,1000):
    	if i % 2 == 0:
    		iput = input1
    	else:
    		iput = input2
    my_htm.set_input_space(iput)
    print(my_htm.input_space.receptors)
    Spatial(my_htm)



