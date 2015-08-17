import csv
import numpy as np
from matplotlib import pyplot as plt
import pprint
import random as r
import itertools as it
import cPickle as pickle
import operator
from scipy import stats
from collections import Counter
# from nupic.research.TP10X2 import TP10X2 as TP
from nupic.research.temporal_memory import TemporalMemory as TP
# from nupic.research.fast_temporal_memory import FastTemporalMemory as TP
from nupic.research.spatial_pooler import SpatialPooler as SP
row_num = 0

retina_radius = 10
image_width = 27 # It's 28x28, but 27 is the last index for the width
left_buffer = retina_radius / 2
right_buffer = left_buffer + (0 if retina_radius % 2 == 0 else 1) # If radius is 5, then right_buffer = 2, but we need 3
right_buffer = 10
left_buffer = 10

def print_num(num_row, n=28, active_char = "*", inactive_char = " ", on_char = "+"):
	counter = 1
	output_string = ''
	for x in num_row:
		output_string += active_char if int(x) > 0 else inactive_char
		if counter % n == 0:
			output_string += '\n'
		counter += 1


	print output_string

def generate_picture(num_encoding):
	num_matrix = np.array(num_encoding).reshape(28,28)
	plt.imshow(num_matrix)
	plt.gray()
	plt.show()

def spatioretinal_encoder(encoding, x, y):
	retina = []
	location = []
	location_matrix = np.zeros((28,28), dtype = np.int)
	num_matrix = np.array(encoding).reshape(28,28)

	for coordinate in it.product(range(x,x+right_buffer),range(y, y+right_buffer)):
		retina.append(num_matrix[coordinate[0]][coordinate[1]])
		location_matrix[coordinate[0]][coordinate[1]] = 1

	location = location_matrix.flatten().tolist()

	return(retina,location)

def generate_patches(patch_width, retina):
	retina_width = int(np.sqrt(len(retina)))
	patches = []
	for i in range(0,retina_width-patch_width+1):
		for j in range(0,retina_width-patch_width+1):
			patch = []
			for k in range(0, patch_width):
				for l in range(0,patch_width):
					patch.append([i+k,j+l])
			patches.append(patch)
			patch = []
	return(patches)

class Synapse:
	def __init__(self, source_input, permanence = 0.5):
		self.source_input = source_input
		self.permanence = permanence

class Column:
	def __init__(self, connected_synapses, cells_per_column = 5, x=0, y=0, hist_num=1000):
		self.connected_synapses = connected_synapses # [s for s in connected_synapses if s.permanence >= .5]
		# self.potential_synapses = [s for s in connected_synapses if s.permanence < .5]
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
		self.connected_perm = .5
		# TP
		# self.cells = [Cell()] * cells_per_column
		self.predicted = 0
		self.predicted_w_strength = 0

def saccade(x,y):
	""" Generate a saccade from position x,y """
	new_x = x + r.choice(range(-10,10))
	new_y = y + r.choice(range(-10,10))
	while new_x not in range(left_buffer, image_width - right_buffer):
		new_x = x + r.choice(range(-10,10))
	while new_y not in range(left_buffer, image_width - right_buffer):
		new_y = y + r.choice(range(-10,10))


	return(new_x,new_y)

def get_indices(vector):
	inds = []
	for i in range(len(vector)):
		if vector[i] == 1:
			inds.append(i)
	return(inds)

def get_guess(cell_hist_dict, cell_inds):
	sums = [0]*10
	for index in cell_inds:
		sums[max(cell_hist_dict[int(index)].iteritems(), key=operator.itemgetter(1))[0]] += 1
		# print(cell_hist_dict[int(index)])

	return sums.index(max(sums))

def convert_binary_vector_to_set(bin_vect):
	""" Assumes string vector because I've been working with strings because I'm an idiot """

	return_set = set()
	for i, j in enumerate(bin_vect):
		if j == 1:
			return_set.add(i)

	return return_set

def convert_set_to_binary_vector(ind_set, n):

	return_binvect = [0] * n
	for index in ind_set:
		return_binvect[index] = 1

	return return_binvect

def hamming_distance(x,y):
    """Calculate the Hamming distance between two bit strings"""
    assert len(x) == len(y)
    count,z = 0,int(x,2)^int(y,2)
    while z:
        count += 1
        z &= z-1 # magic!
    return count

def hamdist(str1, str2):
	diffs = 0
	for ch1, ch2 in zip(str1, str2):
		if ch1 != ch2:
			diffs += 1
	return diffs



# =============================================
# Construct Nums
# =============================================
# answers = []
# num_data = []
# print('*** GETTING NUMS ***')
# with open('train.csv', 'rb') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
#     for row in spamreader:
#         if row_num == 1:
#         	first_num_line = ['1' if int(x) >= 1 else '0' for x in row]

#         if row_num > 0:
#         	answers.append(row.pop(0))
#         	num_data.append(['1' if int(x) >= 1 else '0' for x in row])
        	
#         row_num = row_num + 1
#     first_num_name = first_num_line.pop(0)
#     # generate_picture(first_num_line)
# print('*** GOT NUMS ***')

# pickle.dump(num_data, open( "num_data.p", "wb" ) )
# pickle.dump(answers, open("answers.p", "wb"))

# print('*** GETTING NUMS ***')
# num_data = pickle.load(open("num_data.p", "rb"))
# num_data_np = np.array(num_data, dtype="uint32")
# answers = pickle.load(open("answers.p", "rb"))
# answers_np = np.array(answers, dtype="uint32")
# print('*** GOT NUMS ***')
# np.save("num_data_np",num_data_np)
# np.save("answers_np", answers_np)
# print('*** GETTING NUMS ***')

print('*** GET NUMS ***')

num_data_np = np.load("num_data_np.npy")
answers_np = np.load("answers_np.npy").tolist()

num_data = num_data_np.astype(str)
answers = [str(item) for item in answers_np]

print('*** GOT NUMS ***')

# for i in range(len(num_data)):
# 	print_num(num_data[i],inactive_char='-')
# 	print(answers[i])

# Utility routine for printing the input vector
def formatRow(x):
  s = ''
  for c in range(len(x)):
    if c > 0 and c % 10 == 0:
      s += ' '
    s += str(x[c])
  s += ' '
  return s



# Step 1: create Temporal Pooler instance with appropriate parameters
# tp = TP(columnDimensions=50, cellsPerColumn=4,
#                 initialPerm=0.5, connectedPerm=0.5,
#                 minThreshold=10, newSynapseCount=10,
#                 permanenceInc=0.1, permanenceDec=0.1,
#                 activationThreshold=8,
#                 globalDecay=0, burnIn=1,
#                 checkSynapseConsistency=False,
#                 pamLength=10)

curr_position_pooler = SP(inputDimensions = [28,28],
				   columnDimensions = [5,5],
				   potentialRadius = 50,
				   )

# This definitely needs to be changed as the future saccade should be dependent on the current prediction state
future_position_pooler = SP(inputDimensions = [28,28],
				   columnDimensions = [8,8],
				   potentialRadius = 50,
				   )

retina_pooler = SP(inputDimensions = [10,10],
				   columnDimensions = [10,10])

ret_tp = TP(columnDimensions=(125,),
			cellsPerColumn = 4)

counter = 0
prediction_sex = []
prediction_sex_cells = []
cell_guesses = []
col_guesses = []
# =============================================
# Construct Cell Activation Dictionary
# =============================================

cell_activation_dict = {}
for i in range(ret_tp.columnDimensions[0] * ret_tp.cellsPerColumn):
	cell_activation_dict[i] = {}
	for j in range(10):
		cell_activation_dict[i][j] = 0

# =============================================
# Construct Encodings
# =============================================

def learn_num(number_input, answer, counter):
	first_num_line = number_input
	current_num = answer
	# saccades = [(r.randrange(left_buffer, image_width - right_buffer), r.randrange(left_buffer, image_width - right_buffer)) for x in range(1)]
	retinas = []
	locations = []
	final_encodings = []
	# controlled_sacs = [[4,4],[4,14],[9,4],[9,9],[9,14],[14,4],[14,14]] .32 w/ 32 cells
	controlled_sacs = [[4,4],[4,14],[9,9],[14,4],[14,14]] # .41 w/ 4 cells 
	# controlled_sacs = [[4,4],[9,9],[14,14]] # .36 w/ 4 cells

	# for i in range(500):
	# 	saccades.append(saccade(saccades[i][0],saccades[i][1]))
	for sac in controlled_sacs:
		(retina,location) = spatioretinal_encoder(first_num_line,sac[0],sac[1])
		retinas.append(retina)
		locations.append(location)

	for i in range(len(retinas)):
		final_encodings.append([int(re) for re in retinas[i]]+[int(lo) for lo in locations[i]])

	# =============================================
	# Grab 'inhibition columns'
	# These are connected to 3x3 subgrids of the retina
	# =============================================
	# with open('columns.p', 'r') as infile:
	# 	columns = pickle.load( open( "columns.p", "rb" ) )



	# retina = np.array([int(i) for i in retina]).reshape(10,10)
	# active_columns = [ c for c in columns if sum([ retina[s.source_input[0]][s.source_input[1]] for s in c.connected_synapses] ) in range(2,8)]
	# print_str = ''
	# for c in columns:
	# 	if c in active_columns:
	# 		print_str += '1'
	# 	else:
	# 		print_str += '0'

	# print_num(print_str,n=8,inactive_char='-')
	# col_str = np.array([int(x) for x in print_str]).reshape(8,8)
	# print(col_str)
	# print(sum(sum(col_str)))	
	recent_pred_cells = [[],[],[]]
	guesses = [] 
	prediction_ratio = []
	pred_acc_cols = []
	pred_acc_cells =[]
	curr_active_cols = np.array([0] * 25)
	future_active_cols = curr_active_cols
	retina_active_cols = np.array([0] * 100)
	pred_cols = set()
	active_cols = set()



	# cell_activation_dict = {}
	# for i in range(84864):
	# 	cell_activation_dict[i] = {}
	# 	for j in range(10):
	# 		cell_activation_dict[i][j] = 0



	for i in range(len(retinas)-1):

		# SP the current position
		curr_position_pooler.compute(np.array(locations[i]), True, curr_active_cols)

		# SP the next position
		# future_position_pooler.compute(np.array(locations[i+1]), True, future_active_cols)

		# SP the Retina
		retina_pooler.compute(np.array(retinas[i]), True, retina_active_cols)


		datum = final_encodings[i]
		# iput = convert_binary_vector_to_set(retina_active_cols.tolist()+curr_active_cols.tolist()+future_active_cols.tolist())
		iput = convert_binary_vector_to_set(retina_active_cols.tolist()+curr_active_cols.tolist())
		# iput = convert_binary_vector_to_set(retina_active_cols.tolist())

		preds = ret_tp.predictiveCells
		ret_tp.compute(iput, learn = True)
		actives = ret_tp.activeCells

		# pred_acc_cells = preds.intersection(actives)

		pred_cols = set([ret_tp.columnForCell(x) for x in preds])
		active_cols = set([ret_tp.columnForCell(x) for x in actives])
		pred_acc_cols.append(len(pred_cols.intersection(active_cols)))
		

		# print len(pred_cols),len(active_cols),pred_acc[-1]
		# print iput

		# predicted_ret_vector = ['1' if 1 in x else '0' for x in predictedCells[0:100]]
		# predicted_loc_vector = ['1' if 1 in x else '0' for x in predictedCells[100:884]]

		# print('**** RETINA ****')
		# print_num(retinas[i], n=10, inactive_char='-')

		# print('**** LOC ****')
		# print_num(locations[i], inactive_char='-')

		# print('**** PRED_RETINA ****')
		# print_num(predicted_ret_vector,n=10,inactive_char='-')

		# print('**** PRED_LOC ****')
		# print_num(predicted_loc_vector,inactive_char='-')

	# plt.plot(pred_acc)
	# plt.show()
	min_hamm = 1000
	guess = 100

	# Compute Column Guesses
	if counter > 20:
		for prediction in prediction_sex:
			x = convert_set_to_binary_vector(pred_cols.intersection(active_cols), 125)
			y = convert_set_to_binary_vector(prediction[1], 125)
			d = hamdist(x,y)
			if d < min_hamm:
				min_hamm = d
				guess = prediction[0]

		print current_num,guess,'*** Column Guess ***'

	if current_num == guess:
		col_guesses.append(1)
	else:
		col_guesses.append(0)

	# # Compute Cell Guesses
	# min_hamm = 1000000
	# if counter > 20:
	# 	for prediction in prediction_sex_cells:
	# 		x = convert_set_to_binary_vector(pred_acc_cells, 32*150)
	# 		y = convert_set_to_binary_vector(prediction[1], 32*150)
	# 		d = hamdist(x,y)
	# 		if d < min_hamm:
	# 			min_hamm = d
	# 			guess = prediction[0]

	# 	print current_num,guess,'*** Cell Guess ***'

	# if current_num == guess:
	# 	cell_guesses.append(1)
	# else:
	# 	cell_guesses.append(0)

	prediction_sex.append((current_num,pred_cols.intersection(active_cols)))
	# prediction_sex_cells.append((current_num,pred_acc_cells))

	counter += 1







# sorted_cells = sorted(cell_activation_dict.iteritems(), key=lambda (k,v): (v,k), reverse=False)
# pickle.dump(sorted_cells, open( "1Cells.p", "wb" ) )

# for i in range(len(num_data)):
for i in range(2000):
	# print([str(x) for x in num_data_np[i]])
	# print(num_data[i])
	# print_num(num_data[i], inactive_char='-')
	# print 'The answer is ',answers[i] 
	# print(cell_activation_dict)
	learn_num(num_data[i], answers[i], i)
	ret_tp.reset()

print(float(sum(col_guesses)) / float(len(col_guesses)))
# col_guesses = col_guesses[2500:]
performance = []
for i in range(len(col_guesses)):
	performance.append(float(sum(col_guesses[:i])) / float(i+1))

plt.plot(performance)
plt.show()
# plt.plot(cell_guesses)
# plt.show()


