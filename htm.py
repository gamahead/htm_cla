import random
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
 
if __name__ == '__main__':
    v = [9, 8, 7, 6, 5, 0, 1, 2, 3, 4]
    print([select(v, i) for i in range(10)])

class Synapse:
	def __init__(self, source_input, permanence = 0.5):
		self.source_input = source_input
		self.permanence = permanence

class Column:
	def __init__(self, connected_synapses, x=0, y=0, hist_num=1000):
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


# =============================================
# HTM
# =============================================

columns = []
active_columns = [[]]
t = 0

# =============================================
# Spatial Pooling Algorithms & Parameters
# =============================================

inhibition_radius = 10
desired_local_activity = 10
active_columns[t] = []
connected_perm = .5

def kth_score(columns, desired_local_activity):
	return select([c.overlap for c in columns], desired_local_activity)

def max_duty_cycle(columns):
	return max([c.active_duty_cycle for c in columns])	

# This implementation assumes 2D data because it was built for saccadic vision, 
# so the neighbors are considered the columns in a circular radius 
def neighbors(column):
	x = column.x
	y = column.y
	return([c for c in columns if sqrt((c.x - x)**2 + (c.y - y)**2) <= inhibition_radius])

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

def average_receptive_field_size():

	def receptive_field_size(c):
		sum([1 for s in c.connected_synapses if s >= connected_perm])

	# return( sum( [receptive_field_size(c) for c in columns] ) / len(columns) )
	return(numpy.mean([receptive_field_size(c) for c in columns]))
	

###############################################
# Phase 1: Overlap
###############################################
for c in columns:
	c.overlap = 0

	for s in c.connected_synapses:
		c.overlap += input(t, s.source_input)

	if c.overlap < min_overlap:
		c.overlap = 0
	else:
		c.overlap += c.boost

###############################################
# Phase 2: Inhibition
###############################################
for c in columns:
	min_local_activity = kth_score(neighbors(c), desired_local_activity)

	if c.overlap > 0 and c.overlap >= min_local_activity:
		active_columns[t].append(c)

###############################################
# Phase 3: Learning
###############################################
for c in active_columns[t]:
	
	for s in c.potential_synapses:
		if s.active:
			s.permanence += permanence_inc
			s.permanence = min(1.0, s.permanence)
		else:
			s.permanence -= permanence_dec
			s.permanence = max(0.0, s.permanence)

for c in columns:
	c.min_duty_cycle = 0.01 * max_duty_cycle(c.neighbors)
	c.active_duty_cycle = update_active_duty_cycle(c)
	c.boost = boost_function(c.active_duty_cycle, c.min_duty_cycle)

	c.overlap_duty_cycle = update_overlap_duty_cycle(c)
	if c.overlap_duty_cycle < c.min_duty_cycle:
		increase_permanences(c, 0.1*connected_perm)

inhibition_radius = average_receptive_field_size()

print('done')
