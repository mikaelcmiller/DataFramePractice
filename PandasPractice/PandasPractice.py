import numpy as np
import pandas as pd
import pandas.io.sql as psql
import pyodbc

cnxn = pyodbc.connect("DRIVER={SQL Server};""SERVER=SNADSSQ3;DATABASE=assessorwork;""trusted_connection=yes;")
sql = """SELECT top 100 erijobid,jobdot,jobdottitle FROM sa.fname WHERE jobsource = 'E' and jobactive not in (0,3,5,7,9)"""
job = pd.DataFrame(psql.read_sql(sql, cnxn))
print(job[0:10])
print(job.index)


### Set index to EriJobId when searching (user entry)
job['index1']=job.index
print(job[0:10])
print(job.index)


job['indexsearch']=job['erijobid']
print(job[0:10])
print(job.index)

job.set_index(['index1','indexsearch'],inplace=True)
print(job[0:10])
print(job.index)

# Change 1
print(job.loc[2,:])

#indexset=eval(input("Search ERIJobId: ")) ## Use to search by ERIJobId as index
#print("ERIJobId="+str(indexset),job.loc[indexset,'jobdottitle'],end='\n') ## Use original index for User paging up and down through indicies





# tpl = (1,2,[3,4],"Tuple") # Tuples are immutable
# print(tpl[2])

#import numpy as np
#import pandas as pd
#import pandas.io.sql as psql
#import pyodbc

#cnxn = pyodbc.connect("DRIVER={SQL Server};""SERVER=SNADSSQ3;DATABASE=assessorwork;""trusted_connection=yes;")
#sql = """SELECT erijobid,jobdot,jobdottitle FROM sa.fname WHERE jobsource = 'E' and jobactive not in (0,3,5,7,9)"""
#job = pd.DataFrame(psql.read_sql(sql, cnxn))
##print(job,'\n')
#indexset = 0

#### Set index to EriJobId when searching (user entry)
#job.set_index('erijobid',inplace=True)
#print(job[0:10])
#indexset=eval(input("Search ERIJobId: ")) ## Use to search by ERIJobId as index
#print("ERIJobId="+str(indexset),job.loc[indexset,'jobdottitle'],end='\n') ## Use original index for User paging up and down through indicies

#### Reset index to index, search through original index
### Just for testing purposes, using Search EriJobId for indexing searches
### Will use PgUp/PgDn for indexing +/- 1 (next/last job)
#job.reset_index(inplace=True)
#print(job[0:10])
#indexset=eval(input("Search Index (0-"+str(job.last_valid_index())+"): "))
#print("Index="+str(indexset),job.loc[indexset,'jobdottitle'],end='\n')

#### Next/Last indexing
#while 1:
#    ## Update 'n'/'l' to pgup/pgdown
#    ## Remove from if blocks, change to keypress in class
#    userin = input("Next or Last? (n/l) :")
#    if userin!='n' and userin!='l': break
#    if userin=='n':
#        if indexset==job.last_valid_index(): indexset=0
#        else: indexset=indexset+1
#    else:
#        if indexset==0: indexset=job.last_valid_index()
#        else: indexset=indexset-1
#    print("Index="+str(indexset),job.loc[indexset,'jobdottitle'],end='\n')


## To-Do (not all-inclusive):
## x) Loop index in PgUp/PgDn to account for first/last occurences
## 2) Link to keystroke in self classes (PythonApplication4.py)
## 3) If (user hits PgUp/PgDn):
##    a) If updates: ask user to commit
##    b) If no updates: continue
## 4) Break dataframe into usable parts, link to output sql table for writing



### Random [10][5] Dataframes
#dfrand = pd.DataFrame(np.random.randn(10, 5))
#dfrand2 = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])
#print(dfrand)
#print(dfrand2)



### Using df.loc to add rows to dataframe df
### warning against using ix (it could overwrite incorrect index if index not found)
#df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index= [2.5, 12.6, 4.8], columns=[48, 49, 50])
###!!# There's no index labeled `2`, so you will change the index at position `2`
#df.ix[1] = [60, 50, 40]
#print(df)
#df.ix[1]= [4,5,6] # Resetting to original for next example
### This will make an index labeled `2` and add the new values
#df.loc[1] = [11, 12, 13]
#print(df)



### Setting custom index from a column already in df
### Print out your DataFrame `df` to check it out
#df = pd.DataFrame(data=np.array([[1,2,3],[4,5,6],[7,8,9]]),columns=['A','B','C'])
#print(df)
#print("\nIndex auto: ",df.index.values)
##     A   B   C
## 0   1   2   3
## 1   4   5   6
## 2   7   8   9
## index.values: [0 1 2]
### Set 'C' as the index of your DataFrame
#df = df.set_index('C')
#print("Index('C'): ",df.index.values)
## index.values: [3 6 9]
#print('\nprint(df.loc[6])')
#print(df.loc[6])
## A    4
## B    5
## Name: 6, dtype: int32
#print('\nprint(df.iloc[0])')
#print(df.iloc[0])
## A    1
## B    2
## Name: 3, dtype: int32
#print("\nprint(df.at[6,'B'])") # References set index values (not 0,1,2 indicies)
#print(df.at[6,'B'])
## 5


### Resetting index df.reset_index(level=0, drop=True)
#df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index= [2.5, 12.6, 4.8], columns=[48, 49, 50])
### Check out the weird index of your dataframe
#print(df)
### Inplace=false creates a new column with current index
### Inplace=true seems to act like drop=True
#df = df.reset_index(level=0, inplace=False)
#print(df)
### Use `reset_index()` to reset the values
#df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [4, 5, 6]]), index= [2.5, 12.6, 4.8, 5.2], columns=[48, 49, 50])
#df = df.reset_index(level=0, drop=True)
#print(df)

### Dropping Columns and Handling Duplicates
#df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [4, 5, 6]]), columns=['A', 'B', 'C'])
#print('Print df: ')
#print(df)
#print("Print df.drop('A'...)")
#df.drop('A', axis=1, inplace=True)
#print(df)
### Use drop_duplicates to get rid of duplicate rows (keeping the original)
#print("Print df.drop_duplicates()")
#print(df.drop_duplicates())


### Renaming Columns
#df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [4, 5, 6]]), index=['One','Two','Three','Four'], columns=['A', 'B', 'C'])
### Define the new names of your columns
#newcols = {
#    'A': 'new_column_1',
#    'B': 'new_column_2',
#    'C': 'new_column_3'
#}
### Use `rename()` to rename your columns
#print("Rename df to columns=newcols, replace=True")
#df.rename(columns=newcols, inplace=True)
#print(df)
### Rename your index  # What does this do?
#df.rename(index={1: 'a'})
#print(df)


### Replacing string occurences in df
#df = pd.DataFrame(data=np.array([['OK','Perfecto','Acceptable'],['Awful','Awful','Perfect'],['Acceptable','OK','Poor']]), columns=['Student1','Student2','Student3'])
### Study the DataFrame `df` first
#print(df)


### Replace the strings by numerical values (0-4)
###!!# Perfecto included at [0][1] to test exact match # Exact match is true
#df = df.replace(['Awful', 'Poor', 'OK', 'Acceptable', 'Perfect'], [0, 1, 2, 3, 4])
#print(df)


### Replacing using regex in df
#df = pd.DataFrame(data=np.array([['OK','Perfecto','Acceptable'],['Awfully','Awfulito','Perfectomundo'],['Acceptable','OK','Poor']]), columns=['Student1','Student2','Student3'])
#print(df,'\n')
#df = df.replace({'Perfect.*':'Perfect'}, regex=True).replace({'Awful.*':'Awful'}, regex=True) # df.replace can be strung together
#print(df,'\n')
#df = df.replace(['Awful', 'Poor', 'OK', 'Acceptable', 'Perfect'], [0, 1, 2, 3, 4])
#print(df,'\n')


### Removing Parts from Strings
#df = pd.DataFrame(data=np.array([[1,2,'+3b'],[4,5,'-6B'],[7,8,'+9A']]), columns=['class','test','result'])
#print(df,'\n')
### Delete unwanted parts from the strings in the `result` column
#df['result'] = df['result'].map(lambda x: x.lstrip('+-').rstrip('aAbBcC'))
### Check out the result again
#print(df)


### Practice
#data = np.array([['','ERIJobId','JobDOTTitle'],
#                 ['Row1',1,'Chief Executive Officer'],
#                 ['Row2',2,'Chief Operating Officer']])
#dataframe = pd.DataFrame(data=data[1:,1:],index=data[1:,0],columns=data[0,1:])
#print(dataframe)
#data = np.array([['','ERIJobId','JobDOTTitle'],
#                ['Row1',1,'CEO'],
#                ['Row2',2,'Chief Operating Officer']])
#print(pd.DataFrame(data=data[1:,1:],
#                  index=data[1:,0],
#                  columns=data[0,1:]))


### Dataframe:
###      ERIJobId              JobDOTTitle
### Row1        1  Chief Executive Officer
### Row2        2  Chief Operating Officer


#print(len(dataframe.index)) # 2 # only gives information on height of dataframe
#print(dataframe.shape) # (2,2)
#print('Columns: ',list(dataframe.columns.values)) # prints columns in list format
#print('Indexes: ',list(dataframe.index.values)) # prints index (row) labels in list format


#print(dataframe.iloc[1][1]) # Chief Operating Officer #doesn't include col headers and row indexes
#print(dataframe.loc['Row1']['JobDOTTitle']) # Chief Executive Officer # Must use actual index (ie 'Row1' instead of 0)
#print(dataframe.at['Row1','ERIJobId']) # 1
#print(dataframe.iat[0,1]) # Chief Executive Officer # integer-based indexing
#print(dataframe.get_value('Row2','JobDOTTitle')) # Chief Operating Officer
#print(dataframe.iloc[1])
##  ERIJobId       2
##  JOBDOTTitle    Chief Operating Officer
#print(dataframe.loc[:,'JobDOTTitle']) # Print all rows, column='JobDOTTitle'
##  Row1   Chief Executive Officer
##  Row2   Chief Operating Officer














