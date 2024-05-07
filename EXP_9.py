#!/usr/bin/env python
# coding: utf-8

# In[10]:


pip install pygad


# In[11]:


import pygad
import numpy
import warnings

warnings.filterwarnings('ignore')


# In[13]:


inputs = [4, -2, 3.5, 5, -11, -4.7]
desired_output = 44


# In[14]:


def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution * inputs)
    fitness = 1.0 / numpy.abs(output - desired_output)
    return fitness


# In[15]:


fitness_function = fitness_func
num_generations = 50
num_parents_mating = 4
sol_per_pop = 8
num_genes = len(inputs)
init_range_low = -2
init_range_high = 5
parent_selection_type = "sss"
keep_parents = 1
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10


# In[16]:


ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_function,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    init_range_low=init_range_low,
    init_range_high=init_range_high,
    parent_selection_type=parent_selection_type,
    keep_parents=keep_parents,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    mutation_percent_genes=mutation_percent_genes
)
ga_instance.run()


# In[17]:


solution, solution_fitness, solution_idx = ga_instance.best_solution()


# In[18]:


print("Parameters of the best solution : ", solution)
print("Fitness value of the best solution = ", solution_fitness)


# In[19]:


prediction = numpy.sum(numpy.array(inputs) * solution)
print("Predicted output based on the best solution : ", prediction)


# In[ ]:




