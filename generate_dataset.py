import numpy as np
import pandas as pd
import os

if not os.path.exists('data'):
    os.makedirs('data')

np.random.seed(15)

n_samples = 1000

capital = np.random.uniform(10000, 1000000, n_samples)
monthly_savings = np.random.uniform(500, 20000, n_samples)
age = np.random.uniform(18, 70, n_samples)

labels = np.zeros(n_samples)
for i in range(n_samples):
    score = 0
    if capital[i] > 500000: score += 1
    if monthly_savings[i] > 7000: score += 1
    if 25 <= age[i] <= 50: score += 1
    if score >= 2:
        labels[i] = 1

df = pd.DataFrame({
    'capital': capital,
    'monthly_savings': monthly_savings,
    'age': age,
    'good_partner': labels
})

df.to_csv('data/sample_data.csv', index=False)

print('Dataset generated successfully!')
print(f'Good partners: {labels.sum()}')
print(f'Bad partners: {n_samples - np.sum(labels)}')