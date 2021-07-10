import pandas as pd

df = pd.DataFrame({'name': ['Raphael', 'Donatello'],

                   'mask': ['red', 'purple'],

                   'weapon': ['sai', 'bo staff']})

df.T.to_csv('./results/DC1.csv', index=False)
