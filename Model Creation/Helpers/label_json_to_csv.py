
# coding: utf-8

# In[1]:


import pandas as pd


# In[40]:


for i in range(8):
    path_json = "F:\College\BE\Final year project\FF_Dataset\Kaggle dataset\dfdc_train_part_"+str(i)+"\metadata.json"
    path_csv = "F:\College\BE\Final year project\FF_Dataset\Kaggle dataset\dfdc_train_part_"+str(i)+"\metadata.csv"
    print(path_csv)
    print(path_json)
    read_json = pd.read_json(path_json)
    df = pd.DataFrame(read_json)
    df_2 = pd.DataFrame(df.transpose())
    df_2.to_csv(path_csv)
    read_csv = pd.read_csv(path_csv)
    read_csv.columns = ["URI","label","original","split"]
    read_csv.to_csv(path_csv,index=False)
    print(read_csv.head(5))

