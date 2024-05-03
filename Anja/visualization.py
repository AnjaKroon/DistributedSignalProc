
import matplotlib.pyplot as plt
from math import log
import networkx as nx

'''
PLOTTING CONVERGENCE TIME e(k)
''' 
def plot_single_error(array, name):
    x_values = [x for x, _ in array]
    y_values = [y for _, y in array]
    plt.plot(x_values, y_values)
    plt.xlabel('Transmissions')
    plt.ylabel('||x_k - x_avg||^2')
    plt.title('e(k) vs. Trans. ' + name, fontweight='bold')
    plt.yscale('log') 
    plt.legend(name)
    plt.show()

def plot_multiple_pairs(pairs, plot_name='e(k) vs. Transmissions'):
    for array, name in pairs:
        x_values = [x for x, _ in array]
        y_values = [y for _, y in array]
        plt.plot(x_values, y_values, label=name)
    plt.xlabel('Transmissions')
    plt.ylabel('||x_k - x_avg||^2')
    plt.title(plot_name, fontweight='bold')
    plt.yscale('log') 
    plt.legend([name for _, name in pairs])
    plt.show()

def plot_rgg_side_by_side(rgg1, rgg2, name='Modified Graph'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    pos1 = nx.get_node_attributes(rgg1, 'pos')
    nx.draw(rgg1, pos1, with_labels=True, node_size=300, node_color='skyblue', font_size=12, ax=ax1)
    ax1.axis('equal')
    ax1.set_xlabel('100 km^2')
    ax1.set_ylabel('100 km^2')
    ax1.set_title("Origional Graph")
    
    pos2 = nx.get_node_attributes(rgg2, 'pos')
    nx.draw(rgg2, pos2, with_labels=True, node_size=300, node_color='skyblue', font_size=12, ax=ax2)
    ax2.axis('equal')
    ax2.set_xlabel('100 km^2')
    ax2.set_ylabel('100 km^2')
    ax2.set_title(name)
    
    plt.tight_layout()
    plt.show()

def plot_rgg_nodes(rgg, name='Temperature Sensors'):
    fig, ax = plt.subplots(figsize=(5, 5))
    
    pos = nx.get_node_attributes(rgg, 'pos')
    temp_values = nx.get_node_attributes(rgg, 'temp')
    
    nx.draw_networkx_nodes(rgg, pos, node_size=300, node_color=list(temp_values.values()), cmap='coolwarm', ax=ax)
    nx.draw_networkx_labels(rgg, pos, font_size=12, ax=ax)
    
    ax.set_aspect('equal')  # Set the aspect ratio of the plot to be equal
    ax.set_xlabel('100 km^2')
    ax.set_ylabel('100 km^2')
    ax.set_title(name)
    
    # Add a colorbar to show the temperature gradient
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=min(temp_values.values()), vmax=max(temp_values.values())))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Temperature')
    
    plt.show()


def plot_c_transmissions(array, name, tolerance):
    x_values = [x for x, _ in array]
    y_values = [y for _, y in array]
    plt.plot(x_values, y_values)
    plt.xlabel('c')
    plt.ylabel('Transmissions')
    plt.title('Choice of c for ' + str(name) + ". Tolerance: " + str(tolerance), fontweight='bold')
    plt.show()


