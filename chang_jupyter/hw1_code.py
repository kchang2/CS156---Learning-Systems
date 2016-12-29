import copy
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq


# Parameters
bag1 = ['b','b']
bag2 = ['b','w']

# Question 3
def pick_ball(bag):
	return random.choice(bag)

def pick_bag(bag_set):
	return random.choice(bag_set)

def select(bag_set):
	bag = pick_bag(bag_set)
	return bag, pick_ball(bag)

def run_exercise_3():
	# Question 3
	# probability of 2nd ball black, given 1st ball black
	b_black = 0
	n = 10000000
	for i in xrange(n):
		b1 = copy.deepcopy(bag1)
		b2 = copy.deepcopy(bag2)
		b_set = [b1,b2]
		ball = 'w'

		while ball != 'b':
			bag, ball = select(b_set)

		# get first bag
		bag.remove('b')
		if bag[0] == 'b':
			b_black += 1


	print 'Probability of second ball black: ', float(b_black)/n

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

def run_exercise_5():
	mu = 0.55 # any or more keyword EACH ball probability of being red
	no_red = 1-0.55 # EACH ball probability of being green
	nu = no_red ** 10 # probability of getting no red (or all of 10 balls is green)

	print 'Probability that we get no green balls: ', nu

	p = 1 - (1 - nu)**1000 # 1-nu = probability there is all red
							# 1- above = probability we get at least 1 green

	print 'Probability of at least 1 green (ie. 1 - P(all red)) for 1000 samples: ', p

def assign_output(line, points):
	# aka h(x_)
	output = []
	for pt in points:
		if pt[0]*line[0] + line[1] < pt[1]: # right side
			output.append(-1)
		else:
			output.append(1)	# left side
	return output


def update(w_, x_, y):
	return np.add(w_,np.multiply(y, x_))


def run_exercise_7(n):
	# [(x1, y1), (x2, y2)]
	points = [(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)), (random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))]
	x_coords, y_coords = zip(*points)
	A = vstack([x_coords,ones(len(x_coords))]).T
	m, c = lstsq(A, y_coords)[0]
	print "Line Solution is y = {m}x + {c}".format(m=m,c=c)
	
	pts = gen_uniform_points(n)
	output = assign_output((m,c), pts)



if __name__ == "__main__":
	
	# run_exercise_3()
	# run_exercise_5()
	# run_exercise_7(10)

	n = 100
	d = 2
	x = np.random.rand(d,n)
	print x
	X = ones((d+1,n))
	X[1:,:] = x
	#print X
