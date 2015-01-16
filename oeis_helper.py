from math import log
from random import choice
sequencefile = open('oeis_data/processed.txt', 'r')
sequences = []
for line in sequencefile:
  sequences.append(map(int, line.rstrip().split()))
num_train = 3000
num_test = 1000 #len(sequences) - num_train
counter = 0
def get_next_example():
  global counter, num_train
  if counter >= num_train:
    return None
  sequence = sequences[counter]
  counter += 1
  return sequence

def evaluate(predictor_func):
  global num_train, num_test
  score = 0.0
  for index in xrange(num_train, num_train + num_test):
    #seq = choice(sequences)
    seq = sequences[index]
    predictions = predictor_func(seq[:9])
    assert abs(sum(predictions)-1.0) < 1e-6 and all(map(lambda x : x >= 0.0, predictions)), 'you did not return a probability distribution'
    c_score = log(predictions[seq[9]])
    score += c_score
    #if (c_score < -7):
	#print index + 1 - num_train, c_score, sequences[index][:10]
    if (index+1) % 50 == 0:
      print 'Average score so far: %f (%d examples)' % (score/(index+1-num_train), index+1-num_train)

