"""Network export utilities for various formats."""
import networkx as nx
import json
import numpy as np
from typing import Dict
from pathlib import Path


def export_graphml(graph: nx.DiGraph, filepath: str) -> None:
    """
    Export network to GraphML format (Gephi-compatible).
    
    Args:
        graph: NetworkX graph
        filepath: Output file path
    """
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # GraphML requires string node identifiers
    # Convert all node attributes to strings for compatibility
    graph_copy = graph.copy()
    
    # Convert node attributes
    for node in graph_copy.nodes():
        attrs = graph_copy.nodes[node]
        for key, value in attrs.items():
            if isinstance(value, (list, dict)):
                graph_copy.nodes[node][key] = str(value)
    
    # Convert edge attributes
    for u, v in graph_copy.edges():
        attrs = graph_copy[u][v]
        for key, value in attrs.items():
            if isinstance(value, list):
                graph_copy[u][v][key] = ','.join(str(x) for x in value)
            elif isinstance(value, dict):
                graph_copy[u][v][key] = str(value)
    
    # Write to file
    nx.write_graphml(graph_copy, filepath)
    print(f"Exported GraphML to {filepath}")


def export_gexf(graph: nx.DiGraph, filepath: str) -> None:
    """
    Export network to GEXF format.

    Args:
        graph: NetworkX graph
        filepath: Output file path
    """
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # GEXF requires sanitized attributes (similar to GraphML)
    graph_copy = graph.copy()

    # Convert node attributes to compatible types
    for node in graph_copy.nodes():
        attrs = graph_copy.nodes[node]
        for key, value in attrs.items():
            if isinstance(value, (list, dict)):
                graph_copy.nodes[node][key] = str(value)
            elif isinstance(value, (np.integer, np.floating)):
                graph_copy.nodes[node][key] = float(value)

    # Convert edge attributes to compatible types
    for u, v in graph_copy.edges():
        attrs = graph_copy[u][v]
        for key, value in attrs.items():
            if isinstance(value, list):
                graph_copy[u][v][key] = ','.join(str(x) for x in value)
            elif isinstance(value, dict):
                graph_copy[u][v][key] = str(value)
            elif isinstance(value, (np.integer, np.floating)):
                graph_copy[u][v][key] = float(value)

    # Write to file
    nx.write_gexf(graph_copy, filepath)
    print(f"Exported GEXF to {filepath}")


def export_json(graph: nx.DiGraph, filepath: str) -> None:
    """
    Export network to JSON format (D3.js node-link format).

    Args:
        graph: NetworkX graph
        filepath: Output file path
    """
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Convert to node-link format
    # Try new parameter name first, fall back to default if not supported
    try:
        data = nx.node_link_data(graph, edges="links")
    except TypeError:
        # Older NetworkX versions don't support edges parameter
        data = nx.node_link_data(graph)
        # Rename 'edges' key to 'links' if it exists
        if 'edges' in data:
            data['links'] = data.pop('edges')

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    data = convert_numpy_types(data)

    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Exported JSON to {filepath}")


def export_edgelist(graph: nx.DiGraph, filepath: str) -> None:
    """
    Export network to CSV edge list format.
    
    Args:
        graph: NetworkX graph
        filepath: Output file path
    """
    import csv
    
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'source',
            'target',
            'weight',
            'entity_type',
            'source_posts'
        ])
        
        # Write edges
        for u, v, data in graph.edges(data=True):
            writer.writerow([
                u,
                v,
                data.get('weight', 1),
                data.get('entity_type', ''),
                '|'.join(str(x) for x in data.get('source_posts', []))
            ])
    
    print(f"Exported edge list to {filepath}")


def export_adjacency_matrix(graph: nx.DiGraph, filepath: str) -> None:
    """
    Export network as adjacency matrix CSV.
    
    Args:
        graph: NetworkX graph
        filepath: Output file path
    """
    import pandas as pd
    
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Get adjacency matrix
    adjacency = nx.to_pandas_adjacency(graph, weight='weight')
    
    # Save to CSV
    adjacency.to_csv(filepath)
    print(f"Exported adjacency matrix to {filepath}")


def export_statistics(stats: Dict, filepath: str) -> None:
    """
    Export network statistics to JSON.
    
    Args:
        stats: Statistics dictionary
        filepath: Output file path
    """
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert any non-serializable objects
    stats_clean = {}
    for key, value in stats.items():
        if isinstance(value, (list, tuple)):
            stats_clean[key] = [
                [str(item[0]), float(item[1])] if isinstance(item, tuple) else item
                for item in value
            ]
        else:
            stats_clean[key] = value
    
    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(stats_clean, f, indent=2, ensure_ascii=False)
    
    print(f"Exported statistics to {filepath}")


def export_all_formats(
    graph: nx.DiGraph,
    stats: Dict,
    output_dir: str,
    base_name: str = "network"
) -> Dict[str, str]:
    """
    Export network in all available formats.
    
    Args:
        graph: NetworkX graph
        stats: Statistics dictionary
        output_dir: Output directory path
        base_name: Base name for output files
        
    Returns:
        Dictionary mapping format to filepath
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files = {}
    
    # Export each format
    try:
        graphml_path = str(output_path / f"{base_name}.graphml")
        export_graphml(graph, graphml_path)
        files['graphml'] = graphml_path
    except Exception as e:
        print(f"Error exporting GraphML: {e}")
    
    try:
        gexf_path = str(output_path / f"{base_name}.gexf")
        export_gexf(graph, gexf_path)
        files['gexf'] = gexf_path
    except Exception as e:
        print(f"Error exporting GEXF: {e}")
    
    try:
        json_path = str(output_path / f"{base_name}.json")
        export_json(graph, json_path)
        files['json'] = json_path
    except Exception as e:
        print(f"Error exporting JSON: {e}")
    
    try:
        edgelist_path = str(output_path / f"{base_name}_edgelist.csv")
        export_edgelist(graph, edgelist_path)
        files['edgelist'] = edgelist_path
    except Exception as e:
        print(f"Error exporting edge list: {e}")
    
    try:
        stats_path = str(output_path / f"{base_name}_statistics.json")
        export_statistics(stats, stats_path)
        files['statistics'] = stats_path
    except Exception as e:
        print(f"Error exporting statistics: {e}")
    
    print(f"\nExported {len(files)} files to {output_dir}")
    return files


# Example usage
if __name__ == "__main__":
    # Create example graph
    G = nx.DiGraph()
    
    # Add nodes
    G.add_node("@user1", node_type="author", label="User 1", post_count=5)
    G.add_node("John Smith", node_type="person", label="John Smith", mention_count=3)
    G.add_node("Microsoft", node_type="organization", label="Microsoft", mention_count=2)
    
    # Add edges
    G.add_edge("@user1", "John Smith", weight=2, entity_type="PER", source_posts=["post1", "post2"])
    G.add_edge("@user1", "Microsoft", weight=1, entity_type="ORG", source_posts=["post1"])
    
    # Example statistics
    stats = {
        'total_nodes': 3,
        'total_edges': 2,
        'authors': 1,
        'entities_extracted': 3,
        'top_mentioned': [
            ('John Smith', 3),
            ('Microsoft', 2)
        ]
    }
    
    # Export all formats
    files = export_all_formats(G, stats, "./output", "example_network")
    
    print("\nExported files:")
    for format_name, filepath in files.items():
        print(f"  {format_name}: {filepath}")
