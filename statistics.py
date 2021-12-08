# -*- coding: utf-8 -*-
"""statistics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xS6KJ0AV2J90q_foTlcnUcIv9cQjk1wN
"""

import numpy as np
import pandas as pd
import statistics

df = pd.read_csv("district_data.csv")

#Assemble lists of desired variables
funding_per = []
funding = []
grad_rate = []
students = []

no_outliers_funding_per = []
no_outliers_funding = []
no_outliers_grad_rate = []
no_outliers_students = []

outlier_districts = []

for i,r in df.iterrows():
  funding_per.append(int(r['Cost'])/r['Students'])
  funding.append(int(r['Cost']))
  grad_rate.append(r['Graduation Rate'])
  students.append(r['Students'])
  if r['Graduation Rate'] > 20:
    no_outliers_funding_per.append(int(r['Cost'])/r['Students'])
    no_outliers_funding.append(int(r['Cost']))
    no_outliers_grad_rate.append(r['Graduation Rate'])
    no_outliers_students.append(r['Students'])
  else:
    outlier_districts.append(r['District'])

#descriptive statistics on graduation rate
graduation_rate_mean = statistics.mean(no_outliers_grad_rate)
graduation_rate_med = statistics.median(no_outliers_grad_rate)
graduation_rate_std = statistics.pstdev(no_outliers_grad_rate)
graduation_rate_max = max(no_outliers_grad_rate)
graduation_rate_min = min(no_outliers_grad_rate)
print(graduation_rate_mean)
print(graduation_rate_med)
print(graduation_rate_std)
print(graduation_rate_max)
print(graduation_rate_min)

#descriptive statistics on student population
students_mean = statistics.mean(students)
students_med = statistics.median(students)
students_std = statistics.pstdev(students)
students_max = max(students)
students_min = min(students)
print(students_mean)
print(students_med)
print(students_std)
print(students_max)
print(students_min)

#school district funding
spending_mean = statistics.mean(no_outliers_funding)
spending_med = statistics.median(no_outliers_funding)
spending_std = statistics.pstdev(no_outliers_funding)
spending_max = max(no_outliers_funding)
spending_min = min(no_outliers_funding)
print('\n')
print(spending_mean)
print(spending_med)
print(spending_std)
print(spending_max)
print(spending_min)

#school district funding per student.
spending_stud_mean = statistics.mean(no_outliers_funding_per)
spending_stud_med = statistics.median(no_outliers_funding_per)
spending_stud_std = statistics.pstdev(no_outliers_funding_per)
spending_stud_max = max(no_outliers_funding_per)
spending_stud_min = min(no_outliers_funding_per)
print('\n')
print(spending_stud_mean)
print(spending_stud_med)
print(spending_stud_std)
print(spending_stud_max)
print(spending_stud_min)

df = pd.DataFrame(np.array([['graduation rate', graduation_rate_mean, graduation_rate_med,graduation_rate_std,graduation_rate_max,graduation_rate_min],['student population', students_mean, students_med,students_std,students_max,students_min], ['spending', spending_mean, spending_med, spending_std, spending_max, spending_min], ['spending per student', spending_stud_mean, spending_stud_med, spending_stud_std, spending_stud_max, spending_stud_min]]), columns=['question', 'mean', 'median', 'std', 'max', 'min']);
df.to_csv('statistics.csv')