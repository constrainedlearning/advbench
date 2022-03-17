import plotly.express as px
import plotly.graph_objects as go
import torch
import numpy as np
try:
    import wandb
except ImportError:
    print("Wandb not found, plots will not be logged.")

DELTA_DIMS = ["rot","tx","ty"]

def remove_legend_title(ax, name_dict=None, fontsize=16):
    handles, labels = ax.get_legend_handles_labels()
    if name_dict is not None:
        labels = [name_dict[x] for x in labels]
    ax.legend(handles=handles, labels=labels, fontsize=fontsize)

def adjust_legend_fontsize(ax, fontsize):
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, fontsize=fontsize)

def multicol_legend(ax, ncol=2):
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend.remove()
    ax.legend(handles, labels, ncol=ncol, loc='best')

def tick_density(plot, every=2, mod_val=1, axis='x'):
    ticks = plot.get_yticklabels() if axis == 'y' else plot.get_xticklabels()
    for ind, label in enumerate(ticks):
        if ind % every == mod_val:
            label.set_visible(True)
        else:
            label.set_visible(False)

def plot_perturbed_wandb(deltas, metric, name="loss", wandb_args = {}, plot_mode = "surface"):
    if isinstance(deltas, torch.Tensor):
        deltas = deltas.cpu().numpy()
    if isinstance(metric, torch.Tensor):
        metric = metric.cpu().numpy()
    if len(deltas.shape) == 1:
        deltas = deltas[:, None]
    if deltas.shape[1] == 1:
        deltas = deltas[:, 0]
        data = [[x, y] for (x, y) in zip(deltas.tolist(), metric.tolist())]
        table = wandb.Table(data=data, columns = ["delta", "loss"])
        print("plotting line")
        wandb_dict = {name : wandb.plot.scatter(table, "delta", "loss", title=name)}
        wandb_dict.update(wandb_args)
        wandb.log(wandb_dict)

    elif deltas.shape[1] == 3: # plot using plotly
        if plot_mode == "scatter":
            fig = plot_3d_scatter(deltas, metric, axis_labels=DELTA_DIMS)
            wandb_dict = {f"{name} scatter":fig}
            wandb_dict.update(wandb_args)
            wandb.log(wandb_dict)
        elif plot_mode=="surface":
            num_points = int(round(deltas.shape[0]**(1/deltas.shape[1])))
            #deltas = deltas[[range(0,num_points, num_points) , range(0,num_points, num_points),range(0,num_points, num_points)]]
            unmeshed_deltas = []
            for i in range(3):
                unmeshed_deltas.append(deltas[range(0, num_points**(3-i), num_points**(3-i-1)), i])
            metric = metric.reshape(num_points, num_points, num_points)
            for dim in range(1,3):
                fig = plot_3d_surface(unmeshed_deltas[0], unmeshed_deltas[dim], np.squeeze(np.take(metric, [0], axis=dim)), axis_labels=(DELTA_DIMS[0],DELTA_DIMS[dim], "CE-Loss"))
                wandb_dict = {f"{name} perturbation landscape" : fig}
                wandb_dict.update(wandb_args)
                wandb.log(wandb_dict)
    
    elif deltas.shape[1]>3:
        deltas = np.linalg.norm(deltas, axis=1)
        data = [[x, y] for (x, y) in zip(deltas.tolist(), metric.tolist())]
        table = wandb.Table(data=data, columns = ["delta", "loss"])
        print("plotting line")
        wandb_dict = {name : wandb.plot.scatter(table, "delta", "loss", title=name)}
        wandb_dict.update(wandb_args)
        wandb.log(wandb_dict)           
    else:
        print("plotting not implemented for given perturbation dimension")
        pass


def plot_3d_scatter(deltas, loss, axis_labels=None):
    fig = px.scatter_3d(x=deltas[:,0], y=deltas[:,1], z=deltas[:,2], color=loss)
    if axis_labels is not None:
        fig.update_layout(scene = dict(xaxis_title=axis_labels[0], yaxis_title=axis_labels[1], zaxis_title=axis_labels[2]))
    return fig

def plot_3d_surface(x,y, loss, axis_labels=None):
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=loss)])
    if axis_labels is not None:
        fig.update_layout(scene = dict(xaxis_title=axis_labels[0], yaxis_title=axis_labels[1], zaxis_title=axis_labels[2]))
    return  fig


