import glob
import pandas as pd
from distutils.util import strtobool



directoryPath = './results/'

for file_name in glob.glob(directoryPath + '*.csv'):
    df = pd.read_csv(file_name)
    df = df.replace(',', '', regex=True)
    df = df.replace('\[', '', regex=True)
    df = df.replace('\]', '', regex=True)
    print(file_name)
    df2 = pd.DataFrame(df['0'].str.split().values.tolist())
    num_col = len(df2.columns)



    for i in range(num_col):
        df2.iloc[1][i] = float(df2.iloc[1][i])
        df2.iloc[2][i] = float(df2.iloc[2][i])
        df2.iloc[3][i] = float(df2.iloc[3][i])
        df2.iloc[4][i] = float(df2.iloc[4][i])
        df2.iloc[6][i] = float(df2.iloc[6][i])


        df2.iloc[0][i] = float(df2.iloc[0][i])
        df2.iloc[5][i] = float(df2.iloc[5][i])
        df2.iloc[7][i] = float(df2.iloc[7][i])

        df2.iloc[8][i] = strtobool(df2.iloc[8][i])









    df2.to_csv(file_name)





    '''
    tau = df.iloc[-1][1]
    print('tau: ', tau)
    my_str = tau[:-1]
    my_str = my_str[1:]
    print('mystr: ', my_str)
    tau = float(my_str)

    print('tau: ', type(tau))
    

    new_array = []   


    for i in range(number_columns):

        new_array.append(tau)

    #print(new_array)
    df.iloc[-1][1] = new_array
    print(df)
    df.to_csv(file_name)
    number_columns = len(df.iloc[-1][1])
    print('number of columns_last: ', number_columns)
    '''

