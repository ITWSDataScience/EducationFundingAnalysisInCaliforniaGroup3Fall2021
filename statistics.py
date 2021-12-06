# -*- coding: utf-8 -*-
"""statistics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xS6KJ0AV2J90q_foTlcnUcIv9cQjk1wN
"""

import numpy as np
import pandas as pd

df = pd.read_csv("district_data.csv")
df_expense = pd.read_csv("district_expense_data.csv")
df_school = pd.read_csv("district_school_data.csv")

graduation_rate_mean = df['Graduation Rate'].mean()
graduation_rate_med = df['Graduation Rate'].median()
graduation_rate_std = df['Graduation Rate'].std()
graduation_rate_max = df['Graduation Rate'].max()
graduation_rate_min = df['Graduation Rate'].min()
print(graduation_rate_mean)
print(graduation_rate_med)
print(graduation_rate_std)
print(graduation_rate_max)
print(graduation_rate_min)

for i in df['Cost']:
  i = float(i)

spending_mean = df['Cost'].mean()
spending_med = df['Cost'].median()
spending_std = df['Cost'].std()
spending_max = df['Cost'].max()
spending_min = df['Cost'].min()
print('\n')
print(spending_mean)
print(spending_med)
print(spending_std)
print(spending_max)
print(spending_min)


df['cost_student'] = 0
count = 0
for j in df['Students']:
  if(j == '< 10'):
    j = float(9)
  else:
    j = float(j)
  df.iloc[count, -1] = df['Cost'][count]/j
  count+=1


spending_stud_mean = df['cost_student'].mean()
spending_stud_med = df['cost_student'].median()
spending_stud_std = df['cost_student'].std()
spending_stud_max = df['cost_student'].max()
spending_stud_min = df['cost_student'].min()
print('\n')
print(spending_stud_mean)
print(spending_stud_med)
print(spending_stud_std)
print(spending_stud_max)
print(spending_stud_min)

for i in df_expense[' Current Expense ADA ']:
  i = float(i)
spending_student_mean = df_expense[' Current Expense ADA '].mean()
spending_student_med = df_expense[' Current Expense ADA '].median()
spending_student_std = df_expense[' Current Expense ADA '].std()
spending_student_max = df_expense[' Current Expense ADA '].max()
spending_student_min = df_expense[' Current Expense ADA '].min()
#print('\n')
#print(spending_student_mean)
#print(spending_student_med)
#print(spending_student_std)
#print(spending_student_max)
#print(spending_student_min)