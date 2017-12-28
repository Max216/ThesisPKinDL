import torch
import train as t

def experiment1():
	'''
	Punish shared dimension from hypothesis tp p if entailment (h should be more general -> less info)
	Punish every shared dimension ach todo
	'''



	integrator = t.PKIntegrator(loss_fn, params_fn, res, params)
	return ('experiment_pk_1', integrator)


expt_1 = ('exp1', None)



experiments = dict()
experiments['test'] = expt_1