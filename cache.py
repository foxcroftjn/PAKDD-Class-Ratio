import os
import pickle

def load(filepath, generate=None):
    filepath = os.path.join('cache', f'{filepath}.pickle')
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            contents = f.read()
            return pickle.loads(contents)
    assert generate, 'A generate function must be defined when the cache doesn\'t already contain requested data'
    data = generate()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    return data

def save(data, filepath):
    filepath = os.path.join('cache', f'{filepath}.pickle')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
