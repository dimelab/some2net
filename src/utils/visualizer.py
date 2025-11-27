"""Network visualization with Force Atlas 2 layout."""
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, Tuple, Optional
import numpy as np


try:
    from fa2 import ForceAtlas2
    FA2_AVAILABLE = True
except ImportError:
    FA2_AVAILABLE = False
    print("⚠️  Warning: ForceAtlas2 not available. Install with: pip install fa2")


class NetworkVisualizer:
    """Visualize networks with Force Atlas 2 layout."""
    
    def __init__(self):
        """Initialize visualizer."""
        self.colors = {
            'author': '#1f77b4',      # Blue
            'per': '#ff7f0e',         # Orange  
            'person': '#ff7f0e',      # Orange (alias)
            'loc': '#2ca02c',         # Green
            'location': '#2ca02c',    # Green (alias)
            'org': '#d62728',         # Red
            'organization': '#d62728' # Red (alias)
        }
    
    def compute_force_atlas_layout(
        self,
        graph: nx.DiGraph,
        iterations: int = 50,
        gravity: float = 1.0,
        scale: float = 2.0
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute Force Atlas 2 layout for graph.
        
        Args:
            graph: NetworkX graph
            iterations: Number of layout iterations
            gravity: Gravity strength
            scale: Scale factor for positions
            
        Returns:
            Dictionary mapping node to (x, y) position
        """
        if not FA2_AVAILABLE:
            print("⚠️  ForceAtlas2 not available, falling back to spring layout")
            pos = nx.spring_layout(graph, iterations=50, scale=scale)
            return pos
        
        # Initialize ForceAtlas2
        forceatlas2 = ForceAtlas2(
            outboundAttractionDistribution=True,
            linLogMode=False,
            adjustSizes=False,
            edgeWeightInfluence=1.0,
            jitterTolerance=1.0,
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            scalingRatio=scale,
            strongGravityMode=False,
            gravity=gravity,
            verbose=False
        )
        
        # Compute positions
        print(f"🎨 Computing Force Atlas 2 layout ({iterations} iterations)...")
        pos = forceatlas2.forceatlas2_networkx_layout(
            graph,
            pos=None,
            iterations=iterations
        )
        
        return pos
    
    def create_interactive_plot(
        self,
        graph: nx.DiGraph,
        title: str = "Social Network Visualization",
        width: int = 1000,
        height: int = 800,
        layout_iterations: int = 50
    ) -> go.Figure:
        """
        Create interactive Plotly visualization.
        
        Args:
            graph: NetworkX graph
            title: Plot title
            width: Plot width in pixels
            height: Plot height in pixels
            layout_iterations: Force Atlas iterations
            
        Returns:
            Plotly Figure object
        """
        # Compute layout
        pos = self.compute_force_atlas_layout(
            graph,
            iterations=layout_iterations
        )
        
        # Prepare edge traces
        edge_traces = []
        
        # Group edges by type for different colors
        edge_types = {}
        for u, v, data in graph.edges(data=True):
            edge_type = data.get('entity_type', 'default')
            if edge_type not in edge_types:
                edge_types[edge_type] = []
            edge_types[edge_type].append((u, v, data))
        
        # Create trace for each edge type
        edge_colors = {
            'AUTHOR': '#999',
            'PER': '#ff7f0e',
            'LOC': '#2ca02c',
            'ORG': '#d62728',
            'default': '#999'
        }
        
        for edge_type, edges in edge_types.items():
            edge_x = []
            edge_y = []
            edge_weights = []
            
            for u, v, data in edges:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_weights.append(data.get('weight', 1))
            
            # Average weight for this edge type
            avg_weight = np.mean(edge_weights) if edge_weights else 1
            
            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(
                    width=max(0.5, min(3, avg_weight * 0.5)),
                    color=edge_colors.get(edge_type, '#999')
                ),
                hoverinfo='none',
                mode='lines',
                name=f'{edge_type} edges',
                showlegend=True
            )
            edge_traces.append(edge_trace)
        
        # Prepare node traces (one per node type for coloring)
        node_types = {}
        for node, data in graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            if node_type not in node_types:
                node_types[node_type] = {
                    'nodes': [],
                    'x': [],
                    'y': [],
                    'text': [],
                    'size': []
                }
            
            x, y = pos[node]
            node_types[node_type]['nodes'].append(node)
            node_types[node_type]['x'].append(x)
            node_types[node_type]['y'].append(y)
            
            # Create hover text
            label = data.get('label', node)
            mention_count = data.get('mention_count', 0)
            post_count = data.get('post_count', 0)
            
            hover_text = f"<b>{label}</b><br>"
            hover_text += f"Type: {node_type}<br>"
            if node_type == 'author':
                hover_text += f"Posts: {post_count}<br>"
            hover_text += f"Mentions: {mention_count}"
            
            node_types[node_type]['text'].append(hover_text)
            
            # Node size based on mentions
            size = 10 + min(30, mention_count * 2)
            node_types[node_type]['size'].append(size)
        
        # Create trace for each node type
        node_traces = []
        for node_type, data in node_types.items():
            node_trace = go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='markers',
                hoverinfo='text',
                text=data['text'],
                name=node_type.capitalize(),
                marker=dict(
                    size=data['size'],
                    color=self.colors.get(node_type, '#888'),
                    line=dict(width=1, color='white'),
                    symbol='circle'
                ),
                showlegend=True
            )
            node_traces.append(node_trace)
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + node_traces,
            layout=go.Layout(
                title=dict(
                    text=title,
                    x=0.5,
                    xanchor='center'
                ),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=width,
                height=height,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
        )
        
        return fig
    
    def create_simple_stats_plot(self, stats: Dict) -> go.Figure:
        """
        Create simple bar chart of entity type distribution.
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            Plotly Figure
        """
        categories = ['Authors', 'Persons', 'Locations', 'Organizations']
        values = [
            stats.get('authors', 0),
            stats.get('persons', 0),
            stats.get('locations', 0),
            stats.get('organizations', 0)
        ]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Network Composition",
            xaxis_title="Entity Type",
            yaxis_title="Count",
            height=400,
            showlegend=False
        )
        
        return fig


# Example usage
if __name__ == "__main__":
    # Create example graph
    G = nx.DiGraph()
    
    # Add nodes
    G.add_node("@user1", node_type="author", label="User 1", post_count=5, mention_count=0)
    G.add_node("@user2", node_type="author", label="User 2", post_count=3, mention_count=1)
    G.add_node("John Smith", node_type="per", label="John Smith", mention_count=8)
    G.add_node("Microsoft", node_type="org", label="Microsoft", mention_count=5)
    G.add_node("Copenhagen", node_type="loc", label="Copenhagen", mention_count=4)
    G.add_node("Jane Doe", node_type="per", label="Jane Doe", mention_count=3)
    
    # Add edges
    G.add_edge("@user1", "John Smith", weight=3, entity_type="PER")
    G.add_edge("@user1", "Microsoft", weight=2, entity_type="ORG")
    G.add_edge("@user1", "Copenhagen", weight=2, entity_type="LOC")
    G.add_edge("@user2", "John Smith", weight=5, entity_type="PER")
    G.add_edge("@user2", "Jane Doe", weight=3, entity_type="PER")
    G.add_edge("@user2", "@user1", weight=1, entity_type="AUTHOR")
    
    # Create visualizer
    viz = NetworkVisualizer()
    
    # Create interactive plot
    fig = viz.create_interactive_plot(
        G,
        title="Example Social Network",
        layout_iterations=100
    )
    
    # Show plot (in Jupyter or save to HTML)
    fig.write_html("example_network.html")
    print("📊 Visualization saved to example_network.html")
    
    # Create stats plot
    stats = {
        'authors': 2,
        'persons': 2,
        'locations': 1,
        'organizations': 1
    }
    stats_fig = viz.create_simple_stats_plot(stats)
    stats_fig.write_html("stats.html")
    print("📊 Stats plot saved to stats.html")
