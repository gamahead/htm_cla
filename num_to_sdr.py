import csv
import numpy as np
from matplotlib import pyplot as plt
import pprint
import random as r
import itertools as it
import cPickle as pickle
row_num = 0

retina_radius = 10
image_width = 27 # It's 28x28, but 27 is the last index for the width
left_buffer = retina_radius / 2
right_buffer = left_buffer + (0 if retina_radius % 2 == 0 else 1) # If radius is 5, then right_buffer = 2, but we need 3

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

	for coordinate in it.product(range(x-left_buffer,x+right_buffer),range(y-left_buffer, y+right_buffer)):
		retina.append(num_matrix[coordinate[0]][coordinate[1]])
		location_matrix[coordinate[0]][coordinate[1]] = 1

	location = location_matrix.flatten().tolist()

	print_num(retina, retina_radius, inactive_char = "-")
	print_num(location, inactive_char = "-")


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


with open('train.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if row_num == 2:
        	first_num_line = ['1' if int(x) >= 1 else '0' for x in row]

        row_num = row_num + 1

    first_num_name = first_num_line.pop(0)
    # generate_picture(first_num_line)

print_num(first_num_line)
# print first_num_line
# generate_picture([float(x) for x in first_num_line])
saccades = [(r.randrange(left_buffer, image_width - right_buffer), r.randrange(left_buffer, image_width - right_buffer)) for x in range(10000)]
(retina,location) = spatioretinal_encoder(first_num_line,saccades[0][0],saccades[0][1])
# columns = []
# patches = generate_patches(3,retina)
# print('**** patches generated ****')
# for patch in patches:
# 	# print(patch)
# 	columns.append(Column( [Synapse(s) for s in patch]))
# print('**** columns built ****')
# print('col_num = ',len(columns))

# with open('columns.p', 'w') as outfile:
# 	pickle.dump(columns,outfile)
# print('dumped')
# print('loading')
with open('columns.p', 'r') as infile:
	columns = pickle.load( open( "columns.p", "rb" ) )

print_num(retina, 10, inactive_char = '-')

retina = np.array([int(i) for i in retina]).reshape(10,10)
active_columns = [ c for c in columns if sum([ retina[s.source_input[0]][s.source_input[1]] for s in c.connected_synapses] ) in range(2,8)]
print_str = ''
for c in columns:
	if c in active_columns:
		print_str += '1'
	else:
		print_str += '0'

col_str = np.array([int(x) for x in print_str]).reshape(8,8)
print(col_str)
print(sum(sum(col_str)))

from nupic.research.TP10X2 import TP10X2 as TP

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
tp = TP(numberOfCols=50, cellsPerColumn=2,
                initialPerm=0.5, connectedPerm=0.5,
                minThreshold=10, newSynapseCount=10,
                permanenceInc=0.1, permanenceDec=0.0,
                activationThreshold=8,
                globalDecay=0, burnIn=1,
                checkSynapseConsistency=False,
                pamLength=10)



# Step 2: create input vectors to feed to the temporal pooler. Each input vector
# must be numberOfCols wide. Here we create a simple sequence of 5 vectors
# representing the sequence A -> B -> C -> D -> E
x = np.zeros((5,tp.numberOfCols), dtype="uint32")
x[0,0:10]  = 1   # Input SDR representing "A", corresponding to columns 0-9
x[1,10:20] = 1   # Input SDR representing "B", corresponding to columns 10-19
x[2,20:30] = 1   # Input SDR representing "C", corresponding to columns 20-29
x[3,30:40] = 1   # Input SDR representing "D", corresponding to columns 30-39
x[4,40:50] = 1   # Input SDR representing "E", corresponding to columns 40-49


input1 = np.array([1]*10 + [0]*15, dtype="uint32")
input2 = np.array([0]*15 + [1]*10, dtype="uint32")
for i in range(0,1000):
	if i % 2 == 0:
		iput = input1
	else:
		iput = input2

print_num([str(x) for x in input1.flatten()], n=5, inactive_char='-')
# Step 3: send this simple sequence to the temporal pooler for learning
# We repeat the sequence 10 times
for i in range(10):

  # Send each letter in the sequence in order
  for j in range(5):
  	if j % 2 == 0:
  		iput = input1
  	else:
  		iput = input2
  	tp.compute(iput, enableLearn = True, computeInfOutput = False)


  # The reset command tells the TP that a sequence just ended and essentially
  # zeros out all the states. It is not strictly necessary but it's a bit
  # messier without resets, and the TP learns quicker with resets.
  tp.reset()
  

#######################################################################
#
# Step 3: send the same sequence of vectors and look at predictions made by
# temporal pooler
for j in range(5):
  print "\n\n--------","ABCDE"[j],"-----------"
  if j % 2 == 0:
  	iput = input1
  else:
  	iput = input2
  print "Raw input vector\n",formatRow(iput)
  
  # Send each vector to the TP, with learning turned off
  tp.compute(iput, enableLearn = False, computeInfOutput = True)
  
  # This method prints out the active state of each cell followed by the
  # predicted state of each cell. For convenience the cells are grouped
  # 10 at a time. When there are multiple cells per column the printout
  # is arranged so the cells in a column are stacked together
  #
  # What you should notice is that the columns where active state is 1
  # represent the SDR for the current input pattern and the columns where
  # predicted state is 1 represent the SDR for the next expected pattern
  print "\nAll the active and predicted cells:"
  tp.printStates(printPrevious = False, printLearnState = False)
  
  # tp.getPredictedState() gets the predicted cells.
  # predictedCells[c][i] represents the state of the i'th cell in the c'th
  # column. To see if a column is predicted, we can simply take the OR
  # across all the cells in that column. In numpy we can do this by taking 
  # the max along axis 1.
  print "\n\nThe following columns are predicted by the temporal pooler. This"
  print "should correspond to columns in the *next* item in the sequence."
  predictedCells = tp.getPredictedState()
  print formatRow(predictedCells.max(axis=1).nonzero())

input1 = np.array([1]*10 + [0]*15).reshape((5,5))
input2 = np.array([0]*15 + [1]*10).reshape((5,5))
for i in range(0,1000):
	if i % 2 == 0:
		iput = input1
	else:
		iput = input2

print_num([str(x) for x in input1.flatten()], n=5, inactive_char='-')
# with open('1_100.csv', 'wb') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=',')
#     counter = 1
#     for saccade in saccades:
#     	if counter <= 100: 
#     		spamwriter.writerow(spatioretinal_encoder(first_num_line,saccade[0],saccade[1]))
#     		counter += 1






