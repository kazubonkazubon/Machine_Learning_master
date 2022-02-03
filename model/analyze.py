import glob
import os
from natsort import natsorted
import pandas as pd
import shutil
import numpy as np
import statistics
import openpyxl
from openpyxl.styles.alignment import Alignment
import copy
import datetime


def edit_excel(file_name):
  # read input xlsx
  wb1 = openpyxl.load_workbook(filename=file_name)
  i=0
  while True:
    try:
      ws1 = wb1.worksheets[i]
    except:
      break
    i=i+1
    # set column width
    for col in ws1.columns:
      max_length = 0
      column = col[0].column
      for cell in col:
        if len(str(cell.value)) > max_length:
          max_length = len(str(cell.value))
      adjusted_width = (max_length + 2) * 1.05
      ws1.column_dimensions[column].width = adjusted_width
  wb1.save(file_name)

def analyzer(mode,save_file):
  ALL_SAVE_DIR="/content/drive/MyDrive/analyzed"
  if mode=="val":
    save_dir=ALL_SAVE_DIR+"/csv_files"
    excel_data="aggerate_data.xlsx"
    
  else:
    save_dir=ALL_SAVE_DIR+"/csv_files_test"
    excel_data="aggerate_data_test.xlsx"


  #shutil.rmtree(ALL_SAVE_DIR,ignore_errors=True)
  if not os.path.exists(ALL_SAVE_DIR):
    os.makedirs(ALL_SAVE_DIR)
  shutil.rmtree(save_dir,ignore_errors=True)
  os.makedirs(save_dir)
  all_index=["number","id"]
  with open("/content/drive/MyDrive/functions/written_classes_on_ascending.txt","r") as f:
    data=f.readlines()
    data=list(map(lambda x: x.split("\n")[0],data))
    all_index.extend(data)
  all_index.append("accuracy")
  all_index.append("macro avg")
  all_index.append("weighted avg")
  all_index.append("sorted")
  all_data=pd.DataFrame(index=all_index)
  files=glob.glob("/content/drive/MyDrive/models/*")
  name_list=[]
  data_dict={}
  id_dict={}
  dict_num=0
  for file_names in files:
    name=""
    file_name=file_names+"/args.txt"
    with open(file_name,"r") as f:
      data=f.readlines()
      for i in data:
        if "arch:" in i:
          name=name+i.replace("arch: ","").replace("\n","")
        if "DA:" in i and "True" in i:
          name=name+i.replace("DA: "," DA:").replace("\n","")

      f.close()
    with open(file_names+"/labels.txt") as f:
      name=name+" : "+str(len(f.readlines()))
      f.close()
    num=name_list.count(name)+1
    name_list.append(name)
    data_dict[file_names]=[name,num]
    if not name in id_dict:
      id_dict[name]=dict_num
      dict_num=dict_num+1
  
  if os.path.exists(excel_data):
    os.remove(excel_data)
  for file_names,name in zip(files,name_list):
    file_name=file_names+"/args.txt"
    if mode=="val":
      csvfiles=natsorted(glob.glob(file_names+"/csv_files/*"))
    else:
      csvfiles=natsorted(glob.glob(file_names+"/csv_files_test/*"))
    x=pd.DataFrame()
    file_num=1
    culms_list=[]
    for csv_data in csvfiles:
      if "merge.csv" in csv_data:
        continue
      try:
        df=pd.read_csv(csv_data,index_col=0)
      except:
        continue
      culm_name=["precision","recall","f1_score","support"]
      for i in culm_name:
        data=list(df[i])
        for j in range(len(data)):
          try:
            data[j]=float(data[j])
          except:
            continue

        df[i]=data
      acc=df["f1_score"]

      culms_list.append('試行 : '+str(file_num))
      file_num=file_num+1
      x = pd.concat([x, acc], axis=1, join='outer')


    x.columns = culms_list
    num=len(glob.glob(save_dir+"/*"))+1
    l_1d = x.values.tolist()
    index_list=x.index.tolist()
    ave_list=[]
    for data , index in zip(l_1d,index_list):
      if index=="   ":
        ave_list.append("NaN")
        continue
      data=list(map(lambda x: float(x),data))
      ave_list.append(round(statistics.mean(data),4))    
    x["ave"]=ave_list
    average_data=x["ave"]
    average_data=pd.DataFrame({data_dict[file_names][0]:average_data})
    average_data.loc['number'] = ["NO."+str(data_dict[file_names][1])]
    average_data.loc['sorted'] = [data_dict[file_names][0]+"NO."+str(data_dict[file_names][1])]
    average_data.loc["id"]=[id_dict[data_dict[file_names][0]]]
    all_data = pd.concat([all_data, average_data], axis=1, join='outer')
    excel_file=excel_data
    sheet_name="data_"+str(num)
    
    write_mode="a"
    if not os.path.exists(excel_file):
      write_mode="w"
    with pd.ExcelWriter(excel_file, mode=write_mode) as writer:
      x.to_excel(writer, sheet_name=sheet_name)
    file_data=data_dict[file_names]
    x.to_csv(save_dir+"/data_"+str(num)+".csv")
  
  all_data=all_data.replace("NaN","")
  all_data=all_data.sort_values(by="sorted",axis=1)
  all_data=all_data.drop(["sorted","id"],axis=0)
  all_data=all_data.drop(all_data.index[[-1]])
  with pd.ExcelWriter(excel_file, mode="a") as writer:
    all_data.to_excel(writer, sheet_name="all_data")
  edit_excel(excel_file)
  culm=list(set(all_data.columns.values))
  culm_1=[]
  for i in range(len(culm)):
    a=culm[i].split(":")
    culm_1.append(a[0]+" : class num"+a[1])
  df=pd.DataFrame(index=culm_1,columns=["average","macro avg","support"])
  for i,j in zip(culm,culm_1):
    x=all_data[[i]]
    a=list(x.loc["accuracy",:])
    b=list(x.loc["accuracy",:])
    df.loc[j]=[statistics.mean(a),statistics.mean(b),len(a)]
  if not os.path.exists(save_file):
    os.makedirs(save_file)
  shutil.move(save_dir,save_file)
  shutil.move(excel_data,save_file)
  if os.path.exists(excel_data):
    os.remove(excel_data)
  d_today = datetime.date.today()
  with open(save_file+"/date.txt","w") as f:
    f.write("実行日:"+str(d_today))
  if mode=="val":
    matome_path=save_file+"/matome.xlsx"
  else:
    matome_path=save_file+"/matome_test.xlsx"
  with pd.ExcelWriter(matome_path, mode="w") as writer:
    df.to_excel(writer, sheet_name="all_data")
  edit_excel(matome_path)

def concat(save_file):
  master_file=save_file+"/master.xlsx"
  val_file=save_file+"/matome.xlsx"
  test_file=save_file+"/matome_test.xlsx"
  val_data=pd.read_excel(val_file)
  test_data=pd.read_excel(test_file)
  val_data.rename(columns={'Unnamed: 0': 'model'}, inplace=True)
  test_data.rename(columns={'Unnamed: 0': 'model'}, inplace=True)
  val_data.sort_values('model', inplace=True)
  test_data.sort_values('model', inplace=True)
  val_data.insert(0, 'mode', 'validation')
  test_data.insert(0, 'mode', 'test')
  val_data=pd.concat([val_data,test_data])
  val_data.dropna(how='any', axis=0,inplace=True)
  val_data.reset_index(drop=True,inplace=True)
  with pd.ExcelWriter(master_file, mode="w") as writer:
    val_data.to_excel(writer, sheet_name="all_data")
  edit_excel(master_file)
  os.remove(val_file)
  os.remove(test_file)


  
  pass

def main():
  d_today = datetime.date.today()

  ALL_SAVE_DIR="/content/drive/MyDrive/analyzed"
  save_file=ALL_SAVE_DIR+"/"+str(d_today)
  base=save_file
  i=1
  while True:
    if os.path.exists(save_file):
      save_file=base+"_NO."+str(i)
      i=i+1
    else :
      break
  analyzer("val",save_file)
  analyzer("test",save_file)
  concat(save_file)

def analyze():
  main()

if __name__ == '__main__':
  main()
