import matplotlib.pyplot as plt
import numpy as np
def plot_data(data):
    axis = 0
    for i, (key, values) in enumerate(data.items()):
        if len(np.asarray(values, dtype="object").shape) == 1:
            axis += 1
    fig, axs = plt.subplots(axis, 1, sharex='col', figsize=(55, 25.5))

    i = 0
    for key, values in data.items():
        if len(np.asarray(values, dtype="object").shape) == 1:
            axs[i].plot(values)
            axs[i].set_title(key)
            # import pdb; pdb.set_trace()
            if key == "qp_iter":
                axs[i].text(0.9, 0.9, f"Total qp_iters: {sum(values)}", transform=axs[i].transAxes, ha='right')
                axs[i].text(0.9, 0.8, f"Total sqp_iters: {len(values)}", transform=axs[i].transAxes, ha='right')
            i += 1
    plt.tight_layout()

    plt.show()
    return fig
