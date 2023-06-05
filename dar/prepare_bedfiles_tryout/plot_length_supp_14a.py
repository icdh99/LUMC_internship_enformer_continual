import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('/exports/humgen/idenhond/data/DAR/supp_14a_sorted.bed', sep = '\t', names = ['chr', 'start', 'stop'])
print(df)

df['length'] = df['stop'] - df['start']
print(df)

print(f"Shortest sequence is: {min(df['length'])} and occurs {len(df[df['length'] == min(df['length'])])} times")
print(f"Longest sequence is: {max(df['length'])} and occurs {len(df[df['length'] == max(df['length'])])} times")

print(f"length 200: occurs {len(df[df['length'] == 200])} times")
print(f"length 201: occurs {len(df[df['length'] == 201])} times")
print(f"length 202: occurs {len(df[df['length'] == 202])} times")
print(f"length 203: occurs {len(df[df['length'] == 203])} times")
print(f"length 204: occurs {len(df[df['length'] == 204])} times")

plt.figure()
sns.histplot(data = df, x = 'length')
plt.savefig('supp_14a_lengths.png')

plt.figure()
sns.histplot(data = df, x = 'length', log_scale = True)
plt.savefig('supp_14a_lengths_log.png')



## INTERSECT 
df = pd.read_csv('/exports/humgen/idenhond/data/DAR/intersect_14a_enformertest.bed', sep = '\t', names = ['chr', 'start', 'stop'])
print(df)

df['length'] = df['stop'] - df['start']
print(df.head(20))
print(df['chr'].value_counts())

# print(f"Shortest sequence is: {min(df['length'])} and occurs {len(df[df['length'] == min(df['length'])])} times")
# print(f"Longest sequence is: {max(df['length'])} and occurs {len(df[df['length'] == max(df['length'])])} times")

# print(f"length 200: occurs {len(df[df['length'] == 200])} times")
# print(f"length 201: occurs {len(df[df['length'] == 201])} times")
# print(f"length 202: occurs {len(df[df['length'] == 202])} times")
# print(f"length 203: occurs {len(df[df['length'] == 203])} times")
# print(f"length 204: occurs {len(df[df['length'] == 204])} times")

plt.figure()
sns.histplot(data = df, x = 'length')
plt.savefig('intersect_lengths.png')

plt.figure()
sns.histplot(data = df, x = 'length', log_scale = True)
plt.savefig('intersect_lengths_log.png')