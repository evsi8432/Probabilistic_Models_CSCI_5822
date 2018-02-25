
# coding: utf-8

# In[1]:


# Evan Sidrow
# CSCI 5822
# Homework 0

import itertools
import numpy as np


# In[2]:


# task 0

classes = ['1st','2nd','3rd','crew']
ages = ['child','adult']
genders = ['male','female']
survived = ['yes','no']
keys = itertools.product(*[classes,ages,genders,survived])
hist = {}
jpt = {}

for key in keys:
    hist[key] = 0
    
num_people = 0

with open('titanic.txt') as fp:
    for line in fp:
        num_people += 1
        line = tuple(line.split())
        hist[line] += 1.0

for key in hist:
    jpt[key] = hist[key]/num_people

print(hist)
print(jpt)


# In[3]:


# task 1

prob_death = {}
classifier_death = {}

keys = itertools.product(*[classes,ages,genders])

for key in keys:
    num_survived = hist[key + ('yes',)]
    num_died = hist[key + ('no',)]
    tot = num_died + num_survived
    if tot == 0:
        prob_death[key] = 'NA'
        classifier_death[key] = 'NA'
    else:
        prob = num_died/tot
        prob_death[key] = prob
        classifier_death[key] = (prob > 0.5)
    
print(prob_death) 
print(classifier_death)


# In[11]:


# task 2

class_died = {key: 0 for key in classes}
class_survived = {key: 0 for key in classes}
age_died = {key: 0 for key in ages}
age_survived = {key: 0 for key in ages}
gender_died = {key: 0 for key in genders}
gender_survived = {key: 0 for key in genders}

num_died = 0.0
num_survived = 0.0
keys = itertools.product(*[classes,ages,genders])
for key in keys:
    num_died += hist[key + ('no',)]
    num_survived += hist[key + ('yes',)]

keys = itertools.product(*[classes,ages,genders])
for key in keys:
    died_prob = hist[key + ('no',)]/num_died
    class_died[key[0]] += died_prob
    age_died[key[1]] += died_prob
    gender_died[key[2]] += died_prob
    
    survived_prob = hist[key + ('yes',)]/num_survived
    class_survived[key[0]] += survived_prob
    age_survived[key[1]] += survived_prob
    gender_survived[key[2]] += survived_prob
    
# now find the prior probabilities
prob_died = num_died/(num_died + num_survived)
prob_survived = 1.0 - prob_died

# finally, make our niave bayes classifier
niave_bayes = {}
niave_bayes_class = {}
keys = itertools.product(*[classes,ages,genders])
for key in keys:
    
    death = prob_died *             class_died[key[0]] *             age_died[key[1]] *             gender_died[key[2]]
            
    life =  prob_survived *             class_survived[key[0]] *             age_survived[key[1]] *             gender_survived[key[2]]
            
    death_prob = death/(death+life)
    niave_bayes[key] = death_prob
    niave_bayes_class[key] = (death_prob > 0.5)
    
print(niave_bayes)
print(niave_bayes_class)


# In[10]:


print(class_died)
print(class_survived)

print(age_died)
print(age_survived)

print(gender_died)
print(gender_survived)

print(prob_died)
print(prob_survived)


# In[ ]:




