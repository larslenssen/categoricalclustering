import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def plot_clusters_plotly(df, prototypes=None, title=None):
  # Get the unique clusters
  clusters = df['cluster'].unique()
  clusters.sort()
  cols = df.columns
  number_of_cats = (df.iloc[:, :-1].max().values + 1).astype(np.int64, copy=False)
  row_heights = ((number_of_cats * 0.3 + 0.45)).tolist()

  fig = make_subplots(rows=len(cols)-1, cols=len(clusters), horizontal_spacing=0.005, vertical_spacing=0.02, shared_xaxes=True, subplot_titles=[f"C{cluster}, size = {df[df['cluster'] == cluster].shape[0]}" for cluster in clusters], row_heights = row_heights)
  frames = []

  for i, cluster in enumerate(clusters):
      cluster_color = plt.get_cmap('tab10')(i+1)[:3]
      R, G, B = cluster_color
      rgba_color = f'rgba({int(R * 255)}, {int(G * 255)}, {int(B * 255)}, 1.0)'
      for j, col in enumerate(cols):
          if col != 'cluster':
              # Create a DataFrame that contains the counts in the cluster and in all other clusters
              cluster_counts_df = df[df['cluster'] == cluster][col].value_counts()
              other_clusters_counts_df = df[df['cluster'] != cluster][col].value_counts()
              # Combine the two dataframes for stacked bar plot
              combined_df = pd.concat([ cluster_counts_df, other_clusters_counts_df], axis=1)
              combined_df.columns = ['This Cluster', 'Other Clusters']
              combined_df = combined_df.fillna(0)
              # Sort the DataFrame by index (y-axis values) before plotting
              combined_df.sort_index(inplace=True)
              # Replace the index with the column names
              combined_df.index = col.split('|') if '|' in col else ('non-' + col, col)
              # Plot the counts in all other clusters and in the cluster
              fig.add_trace(go.Bar(y=combined_df.index, x=combined_df['This Cluster'], name='This Cluster', orientation='h', marker_color=rgba_color), row=j+1, col=i+1)
              fig.add_trace(go.Bar(y=combined_df.index, x=combined_df['Other Clusters'], name='Other Clusters', orientation='h', marker_color='silver'), row=j+1, col=i+1)
              if i == 0:
                fig.update_yaxes(tickmode='array', tickvals=combined_df.index, row=j+1, col=i+1)
              else:
                fig.update_yaxes(showticklabels=False, row=j+1, col=i+1)
              # Highlight the first y-tick ## FIXME comment
              if prototypes is not None:
                fig.add_annotation(
                    xref='paper',
                    yref='y',
                    x=0,
                    y=prototypes[i,j],
                    text=' P ',
                    xanchor='right',
                    showarrow=False,
                    borderwidth=0, borderpad=0, height=10,
                    font=dict(color='black', size=10),
                    bgcolor='rgba(0,0,0,0)',
                    row=j+1,
                    col=i+1
                )
  fig.update_layout(barmode='stack', showlegend=False, height=60*len(number_of_cats), bargap=0.1)
  if title is not None:
    fig.update_layout(title_text=title)
  fig.show()

def plot_dendogram(matrix, name, num_clusters=None):
  plt.figure(figsize=(10, 7))
  dendro = dendrogram(matrix, color_threshold=matrix[-num_clusters+1, 2] if num_clusters else None)
  plt.title(name)
  plt.xlabel('Sample index')
  plt.ylabel('Distance')
  # Annotate the dendrogram with numbers
  max_d1_by_color, best_values_by_color = {}, {}
  maxdist = 0
  for i, d, c in zip(dendro['icoord'], dendro['dcoord'], dendro['color_list']):
      x, y = i[1], d[1]
      if y > maxdist: maxdist = y
    # Check if the color code is already in the dictionary
      if c not in max_d1_by_color:
          max_d1_by_color[c] = y
          best_values_by_color[c] = {'x': x, 'y': y}
      else:
        # Update the maximum d[1] and corresponding x, y values if the current value is higher
          if y > max_d1_by_color[c]:
              max_d1_by_color[c] = y
              best_values_by_color[c] = {'x': x, 'y': y}
  for color, max_d1 in max_d1_by_color.items():
    if color != 'C0':
        best_values = best_values_by_color[color]
        plt.text(best_values['x'], best_values['y'] + 0.01*maxdist, f"{color}", va='center', ha='center')
  plt.show()

#@jit(nopython=True)
def calculate_counts_categorical(X, bin_max):
  # X = X.reshape(1, -1).astype(np.int64, copy=False) if X.ndim == 1 else X.astype(np.int64, copy=False)
  if X.ndim == 1: X = X.copy().reshape(1, -1)
  counts = np.zeros((bin_max + 1, X.shape[1]), dtype=np.int64)
  for i in range(X.shape[1]):
      counts[:np.bincount(X[:, i]).size, i] = np.bincount(X[:, i])
  return counts


def get_prototypes(data, number_of_cluster, p, weights):
  max_value = np.amax(data)
  counts_overall = calculate_counts_categorical(data[:, :-1], max_value)
  # Create a 2D numpy array of zeros
  cluster_counts = np.zeros((number_of_cluster, max_value + 1, data.shape[1]), dtype=np.int64)
  cluster_index = data.shape[1] - 1
  # Set the column corresponding to the cluster of each point to 1
  for i in range(len(data)):
      cluster_counts[data[i, cluster_index] - 1] += np.hstack(
          (np.zeros((max_value + 1, 1), dtype=np.int64), calculate_counts_categorical(data[i, :-1], max_value)))
      cluster_counts[data[i, cluster_index] - 1, 0, 0] += 1
  # Initialize prototypes
  prototypes = np.zeros((number_of_cluster, data.shape[1] - 1), dtype=np.int64)
  # calculate for each ccluster
  for i in range(number_of_cluster):
      counts_rest = counts_overall - cluster_counts[i, :, 1:]
      counts_cluster = cluster_counts[i, :, 1:] / cluster_counts[i, 0, 0]
      counts_rest = counts_rest / (len(data) - cluster_counts[i, 0, 0])
      counts_dif = counts_cluster - counts_rest
      prototypes[i] = counts_dif.argmax(axis=0)
  return prototypes

def categorical_cqm(data, number_of_cluster, p, weights):
    max_value = np.amax(data)
    # Create a 2D numpy array of zeros
    cluster_counts = np.zeros((number_of_cluster, max_value+1, data.shape[1]), dtype=np.int64)
    cluster_index = data.shape[1] - 1
    # Set the column corresponding to the cluster of each point to 1
    for i in range(len(data)):
      cluster_counts[data[i,cluster_index]-1] += np.hstack((np.zeros((max_value+1, 1), dtype=np.int64), calculate_counts_categorical(data[i,:-1], max_value)))
      cluster_counts[data[i,cluster_index]-1, 0, 0] += 1
    # Initialize the cluster quality value
    cqv = 0
    # calculate for each ccluster
    cqv_clust_best = float('-inf')
    for i in range(number_of_cluster):
        cqv_clust_own = cat_sim(cluster_counts[i], Y=None, p=p, weights=weights)
        cqv_clust_best = float('-inf')
        for j in range(number_of_cluster):
          if i != j:
            cqv_clust_tmp = cat_sim(cluster_counts[i] + cluster_counts[j], Y=None, p=p, weights=weights)
            if cqv_clust_tmp > cqv_clust_best: cqv_clust_best = cqv_clust_tmp
        cqv += (cluster_counts[i, 0, 0] / len(data) )  * (cqv_clust_own - cqv_clust_best)
        # + cluster_counts[i,0,0])
    return cqv

def cat_sim(X, Y, p, weights):
    XYcounts = X
    XYlen = X[0,0]
    if Y is not None:
      XYcounts = X + Y
      XYlen = X[0,0] + Y[0,0]
    d =  0
    for i in range(1, XYcounts.shape[1]):
        for j in range(len(XYcounts)):
            if p[j, i-1] == 0.0 or XYlen: continue # never in the data set
            p_i = XYcounts[j,i] / XYlen
            if p_i > 0:
              p_d = p[j, i-1]
              d += weights[i-1] * p_i * np.log2(p_i  / p_d)
    return d

