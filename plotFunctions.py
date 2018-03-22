#simple plotting macro that uses a heatmap color scheme and avoids interpolation
def plot_heatmap(data, outfile):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.set_xlabel('Plane')
    ax.set_ylabel('Cell')

    imgplot = plt.imshow(np.rot90(data), cmap='gist_heat_r', interpolation='none', extent=[0,100,0,80])

    ax.imshow(np.rot90(data), cmap='gist_heat_r', interpolation='none', extent=[0,100,0,80])
    ax.imshow(np.rot90(data), cmap='gist_heat_r', interpolation='none', extent=[0,100,0,80])
    plt.colorbar()

    plt.savefig(outfile,dpi = 1000)

#edits to color style to make the story clearer on a large projector
def plot_heatmap_presentation(data, outfile):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.set_xlabel('Plane')
    ax.set_ylabel('Cell')

    imgplot = plt.imshow(np.rot90(data), vmin=-0.1, vmax=0.1, cmap='RdGy', interpolation='none', extent=[0,100,0,80])

    ax.imshow(np.rot90(data), vmin=-0.1, vmax=0.1, cmap='RdGy', interpolation='none', extent=[0,100,0,80])
    ax.imshow(np.rot90(data), vmin=-0.1, vmax=0.1, cmap='RdGy', interpolation='none', extent=[0,100,0,80])
    bar=plt.colorbar()
    bar.set_ticks([])
    plt.savefig(outfile,dpi = 1000)
