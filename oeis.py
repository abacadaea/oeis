import math
from math import log
from random import random
from oeis_helper import get_next_example, evaluate
from collections import defaultdict

def super_dict ():
    return defaultdict (super_dict)

DIR_FACTOR = .7
NDIR_FACTOR = .93
ERR_EXP = .58
BASELINE = 2e-5
SIZE = 9
PAR_LEN_PARAM = 9

INF = 1e9

first = []
last = []

parity_count = dict ()

num_train = 0
while True:
    example = get_next_example()
    if (example == None):
	break
    num_train += 1
    first.append (example[0:9])
    last.append (example [9])

#BEGIN PARITY
def get_parity (train):
    ret = 0
    for i in range (len (train)):
	ret = ret + ((train [i] % 2) << i)
    return ret

def train_parity ():
    for i in range (1 << PAR_LEN_PARAM):
	parity_count[i] = [1,1]
    for i in range (num_train):
	mask = get_parity (first [i][-PAR_LEN_PARAM:])
	parity_count[mask][last[i] % 2] += 1
    

def parity_distrib (dist, example):
    counts = parity_count [get_parity (example[-PAR_LEN_PARAM:])]
    p = [float (counts [0]) / (counts[0] + counts[1]), float (counts[1]) / (counts[0] + counts[1])]
    for i in range (1000):
	dist [i] *= p[i % 2]

    dist = normalize (dist)
    return dist

#END PARITY

	
def pad_distrib (dist):
    for i in range (1000):
	dist[i] *= 1 - 1e3 * BASELINE
    for i in range (1000):
	dist[i] += BASELINE
    return dist

def get_gaussian (mean, sig, is_mono):
    if (mean < 0):
	mean = 1
    if (mean > 999):
	mean = 950
    

    ret = []
    R = 2 * (sig ** 2)
    for i in range (1000):
	x = (mean - i) ** 2.0 / R
	ret.append (math.exp (-x))

    
    if (is_mono != None):
	if (is_mono [1] == 1):
	    for i in range (is_mono[0]):
		ret[i] = 1e-5
	else:
	    for i in range (is_mono[0] + 1, 1000):
		ret[i] = 1e-5

    return normalize (ret)

def normalize (dist):
    S = sum (dist)
    for i in range (1000):
	dist [i] /= S
    return dist


def magn (arr):
    ret = 0.0
    for a in arr:
	ret += a ** 2
    return ret ** 0.5

def poly_eval (poly, x, deg):
    ret = 0.0
    for i in range (deg + 1):
	ret += (x ** i) * poly[i]
    return ret

def error (example, poly, deg):
    err = 0
    for i in range (SIZE):
	err += pow (poly_eval (poly, i, deg) - example[i], 2)
    return err

def poly (example, deg):
    ret = [0] * (deg + 1)
    
    alpha = pow (10, -deg)
    prev_err = INF
    n_iter = 0

    while n_iter < 40000:
	n_iter += 1

	delta = [0] * (deg + 1)
	for i in range (deg + 1):
	    for j in range (SIZE):
		delta [i] += 2.0 * (j**i) * (poly_eval (ret, j, deg) - example[j])

	for i in range (deg + 1):
	    ret [i] -= alpha * delta[i]

	cur_err = error (example, ret, deg)
	if (magn (delta) < .001):
	    break
	
	if (cur_err > 1.2 * prev_err and alpha > .01 * pow (10, -deg)):
	    alpha /= 2
	prev_err = cur_err

    return [ret,cur_err]

def monotonic (example, sz):
    down = True
    up = True
    for i in range (sz - 1):
	if (example[i] <= example[i + 1]):
	    down = False
	if (example[i] >= example[i + 1]):
	    up = False
    #return (not down or not up)
    if up:
	return 1
    elif down:
	return -1
    else:
	return 0

def convex (example):
    f_diff = []
    for i in range (SIZE - 1):
	f_diff.append (example [i + 1] - example [i])
    return monotonic (f_diff, SIZE - 1)

#std based on mean

def get_distrib (example, deg, FACTOR, cutoff, direct):
    p, err = poly (example, deg)
    mu = poly_eval (p, SIZE, deg)
    sig = max (FACTOR * (err ** (ERR_EXP)), FACTOR) #best for linear

    if num_iter < 0:
	print num_iter, num_mono
	print example
	print cutoff
	print round (mu), sig, err
	print p, "\n"

    if deg == 1 and convex (example):
	if direct == 1:
	    if mu < example[SIZE - 1]:
		return get_distrib (example, 2, 2 * FACTOR, cutoff, direct)
	if direct == -1:
	    if mu > example[SIZE - 1]:
		return get_distrib (example, 2, 2 * FACTOR, cutoff, direct)

    ret = get_gaussian (mu, sig, cutoff)
    ret = parity_distrib (ret, example)
    ret = pad_distrib (ret)
    return ret


num_iter = 0
num_mono = 0

def predictor_func(example):
    example = example[-SIZE:]
    default = [1.0/1000] * 1000

    global num_iter
    num_iter += 1

    direct = monotonic (example, SIZE)

    if direct != 0:
	global num_mono
	num_mono += 1
	cutoff = [example [SIZE - 1], direct]
	return get_distrib (example, 1, DIR_FACTOR, cutoff, direct)
    else:
	return get_distrib (example, 1, NDIR_FACTOR, None, direct)

train_parity ()
#for x in parity_count:
    #print x, parity_count[x] 

print "Parity Trained"

evaluate(predictor_func)
