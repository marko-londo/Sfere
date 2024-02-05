import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from itertools import combinations

exploded_df = pd.read_csv(r"C:\Users\londo\01\001\Repos\Sfere\Notebooks\03-Feature-Extraction\entities_graph_clean.csv")

def filter_entities(entities_list):
    return {(entity, type) for entity, type in entities_list if not (entity == "Earth" and type == "LOC")}

entity_types = [
    'Person',
    'Location',
    'Facility',
    'Nationality or Religious or Political group',
]
colors = [
    "#583bac",
    "#5080c0",
    "#66d2c4",
    "#9be78e"
]

color_map = dict(zip(entity_types, colors))

G = nx.Graph()

for _, row in exploded_df.iterrows():
    entity = row['Entity']
    entity_type = row['Type']
    G.add_node(entity, type=entity_type)

for title, group in exploded_df.groupby('Text'):
    entities = group['Entity'].tolist()
    for source, target in combinations(entities, 2):
        G.add_edge(source, target)

pos = nx.kamada_kawai_layout(G)

edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#837d94'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
node_color = []
node_text = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_color.append(color_map.get(G.nodes[node]['type'], 'grey'))
    node_text.append(node)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    hoverinfo='text',
    text=node_text,
    textposition="top center",
    marker=dict(
        size=10,
        color=node_color,
        line_width=2))

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
fig.update_layout(
    width=1200,
    height=800)

def show():
    st.write('#### Network graph of entities in "The Dying Earth"')
    for entity_type, color in color_map.items():
        st.markdown(f"<span style='display:inline-block; width:12px; height:12px; margin-right:5px; background-color:{color};'></span>{entity_type}", unsafe_allow_html=True)
    st.plotly_chart(fig)