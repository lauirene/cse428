
import pickle
def load_pickle(path):
    with open(path,'rb') as file:
        data=pickle.load(file)
    return data

def write_pickle(data,path):
    with open(path,'wb') as file:
        pickle.dump(data, file)

