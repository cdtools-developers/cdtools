import cdtools
from matplotlib import pyplot as plt

model = cdtools.models.FancyPtycho.from_results_h5(
    'example_reconstructions/gold_balls.h5',
    obj_view_crop=260, # How far in to crop from the edge
    units='um', # The units to display in
)

model.inspect()
plt.show()
