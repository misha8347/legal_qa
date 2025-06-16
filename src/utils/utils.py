import pickle

def save_obj(obj, name):
    pickle.dump(obj,open(name + '.pkl', 'wb'), protocol=4)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)