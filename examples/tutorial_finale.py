from tutorial_basic_ptycho_dataset import BasicPtychoDataset
from tutorial_simple_ptycho import SimplePtycho
import torch as t
from h5py import File
from matplotlib import pyplot as plt

filename = 'example_data/lab_ptycho_data.cxi'
with File(filename, 'r') as f:
    dataset = BasicPtychoDataset.from_cxi(f)    

dataset.inspect()

model = SimplePtycho.from_dataset(dataset)

if t.cuda.is_available():
    model.to(device='cuda')
    dataset.get_as(device='cuda')

for loss in model.Adam_optimize(10, dataset):
    model.inspect(dataset)
    print(model.report())

model.inspect(dataset)
model.compare(dataset)
plt.show()
