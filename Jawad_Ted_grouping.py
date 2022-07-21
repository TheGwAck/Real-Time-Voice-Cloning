from genericpath import isdir
import sys
from pathlib import Path
import pathlib
path = Path('/content/drive/MyDrive/Collabera_William/testing_datasets/TEDLIUM_release-3/Tedlium_Conversion_Testing')
all_objs = path.glob('**/*')
files = [f for f in all_objs if f.is_file()]
a= list(path.iterdir())

for file in files:
  temp = file.stem
  x = temp.split('_')
  pathtocheck= Path(file.parents[0],x[0])
  print(pathtocheck)
  if pathtocheck in a:
    print("jj")
    file.rename(pathlib.PurePath(str(file.parents[0]),x[0], Path(file.name))) 
  else:
     newpath = pathlib.PurePath(str(file.parents[0]) , x[0])
     Path(newpath).mkdir(parents=True, exist_ok=True)
