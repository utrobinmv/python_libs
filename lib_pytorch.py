import torch

def torch_vector_to_numpy(vector):
    return vector.numpy()
    
def numpy_array_to_torch_vector(data):
    return torch.from_numpy(data)
    

