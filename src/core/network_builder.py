"""
Network Builder Module

Constructs directed networks from social media posts and extracted entities.
Creates nodes for authors and entities, edges for mentions, and tracks statistics.
"""

import networkx as nx
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import logging
from collections import Counter

from .entity_resolver import EntityResolver

logger = logging.getLogger(__name__)


class NetworkBuilder:
    """
    Build directed networks from social media posts and named entities.

    Network Structure:
    - Nodes: Authors and entities (PER, LOC, ORG)
    - Edges: Directed from author to mentioned entity
    - Edge weights: Frequency of mentions
    - Special: Author-to-author edges when authors mention each other
    """

    def __init__(
        self,
        use_entity_resolver: bool = True,
        create_author_edges: bool = True
    ):
        """
        Initialize network builder.

        Args:
            use_entity_resolver: Whether to use entity resolver for deduplication
            create_author_edges: Whether to create author-to-author edges
        """
        self.graph = nx.DiGraph()
        self.use_entity_resolver = use_entity_resolver
        self.create_author_edges = create_author_edges

        # Initialize entity resolver if enabled
        if use_entity_resolver:
            self.entity_resolver = EntityResolver()
        else:
            self.entity_resolver = None

        # Track statistics
        self.stats = {
            'posts_processed': 0,
            'entities_added': 0,
            'edges_created': 0,
            'author_mentions': 0
        }

        # Track all authors for author-to-author matching
        self.authors: Set[str] = set()

    def add_post(
        self,
        author: str,
        entities: List[Dict],
        post_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        node_metadata: Optional[Dict] = None,
        edge_metadata: Optional[Dict] = None
    ):
        """
        Add a post to the network.

        Args:
            author: Author name/handle
            entities: List of entity dictionaries from NER
                     Each should have: {'text': str, 'type': str, 'score': float}
            post_id: Optional post identifier
            timestamp: Optional timestamp string
            node_metadata: Optional metadata to attach to author node
            edge_metadata: Optional metadata to attach to edges
        """
        self.stats['posts_processed'] += 1

        # Normalize author name
        author = str(author).strip()
        if not author:
            logger.warning("Empty author name, skipping post")
            return

        # Track author
        self.authors.add(author)

        # Add author node if not exists
        if not self.graph.has_node(author):
            node_attrs = {
                'node_type': 'author',
                'label': author,
                'mention_count': 0,
                'post_count': 0
            }

            # Add node metadata if provided
            if node_metadata:
                # Prefix metadata keys to avoid conflicts
                for key, value in node_metadata.items():
                    if key not in node_attrs:  # Don't override core attributes
                        node_attrs[f'meta_{key}'] = value

            self.graph.add_node(author, **node_attrs)
        else:
            # Update existing node with metadata if provided
            if node_metadata:
                for key, value in node_metadata.items():
                    meta_key = f'meta_{key}'
                    # Only add if not already present (first occurrence wins)
                    if meta_key not in self.graph.nodes[author]:
                        self.graph.nodes[author][meta_key] = value

        # Ensure post_count attribute exists (for backwards compatibility)
        if 'post_count' not in self.graph.nodes[author]:
            self.graph.nodes[author]['post_count'] = 0

        # Increment author's post count
        self.graph.nodes[author]['post_count'] += 1

        # Process each entity
        for entity in entities:
            entity_text = entity.get('text', '').strip()
            entity_type = entity.get('type', 'UNKNOWN')
            entity_score = entity.get('score', 0.0)

            # Entity linking metadata (if available)
            wikidata_id = entity.get('wikidata_id')
            wikipedia_url = entity.get('wikipedia_url')
            canonical_name = entity.get('canonical_name')
            is_linked = entity.get('is_linked', False)

            if not entity_text:
                continue

            # Resolve entity to canonical form if resolver enabled
            if self.entity_resolver:
                # Use Wikidata ID for enhanced resolution if available
                canonical_entity = self.entity_resolver.get_canonical_form(
                    entity_text,
                    wikidata_id=wikidata_id,
                    canonical_name=canonical_name
                )
            else:
                canonical_entity = entity_text

            # Check if entity is an author mention
            is_author_mention = False
            if self.create_author_edges and self.entity_resolver:
                # Check against all known authors
                for known_author in self.authors:
                    if self.entity_resolver.is_author_mention(known_author, entity_text):
                        # This is an author mention
                        is_author_mention = True
                        self._add_author_mention_edge(
                            author, known_author, post_id, timestamp
                        )
                        self.stats['author_mentions'] += 1
                        break

            # Add entity node and edge if not an author mention
            # (or if we're not creating author edges)
            if not is_author_mention or not self.create_author_edges:
                # Pass entity linking metadata
                entity_metadata = {
                    'wikidata_id': wikidata_id,
                    'wikipedia_url': wikipedia_url,
                    'is_linked': is_linked
                } if is_linked else {}

                self._add_entity_edge(
                    author,
                    canonical_entity,
                    entity_type,
                    entity_score,
                    post_id,
                    timestamp,
                    entity_metadata,
                    edge_metadata
                )
                self.stats['entities_added'] += 1

    def _add_entity_edge(
        self,
        author: str,
        entity: str,
        entity_type: str,
        score: float,
        post_id: Optional[str],
        timestamp: Optional[str],
        entity_metadata: Optional[Dict] = None,
        edge_metadata: Optional[Dict] = None
    ):
        """Add or update edge from author to entity with optional metadata."""

        # Map entity type to node type
        node_type_map = {
            'PER': 'person',
            'LOC': 'location',
            'ORG': 'organization',
            'PERSON': 'person',
            'LOCATION': 'location',
            'ORGANIZATION': 'organization'
        }
        node_type = node_type_map.get(entity_type.upper(), 'entity')

        # Add entity node if not exists
        if not self.graph.has_node(entity):
            node_attrs = {
                'node_type': node_type,
                'label': entity,
                'mention_count': 0
            }

            # Add entity linking metadata if available
            if entity_metadata:
                if entity_metadata.get('wikidata_id'):
                    node_attrs['wikidata_id'] = entity_metadata['wikidata_id']
                if entity_metadata.get('wikipedia_url'):
                    node_attrs['wikipedia_url'] = entity_metadata['wikipedia_url']
                if entity_metadata.get('is_linked'):
                    node_attrs['is_linked'] = True

            self.graph.add_node(entity, **node_attrs)
        else:
            # Update existing node with entity linking metadata if available
            if entity_metadata:
                if entity_metadata.get('wikidata_id') and 'wikidata_id' not in self.graph.nodes[entity]:
                    self.graph.nodes[entity]['wikidata_id'] = entity_metadata['wikidata_id']
                if entity_metadata.get('wikipedia_url') and 'wikipedia_url' not in self.graph.nodes[entity]:
                    self.graph.nodes[entity]['wikipedia_url'] = entity_metadata['wikipedia_url']
                if entity_metadata.get('is_linked') and 'is_linked' not in self.graph.nodes[entity]:
                    self.graph.nodes[entity]['is_linked'] = True

        # Increment mention count
        self.graph.nodes[entity]['mention_count'] += 1

        # Add or update edge
        if self.graph.has_edge(author, entity):
            # Edge exists, increment weight
            self.graph[author][entity]['weight'] += 1

            # Add post_id to source_posts if provided
            if post_id:
                if 'source_posts' not in self.graph[author][entity]:
                    self.graph[author][entity]['source_posts'] = []
                self.graph[author][entity]['source_posts'].append(post_id)

            # Update last_mention timestamp
            if timestamp:
                self.graph[author][entity]['last_mention'] = timestamp

            # Add edge metadata if provided
            if edge_metadata:
                for key, value in edge_metadata.items():
                    meta_key = f'meta_{key}'
                    if meta_key not in self.graph[author][entity]:
                        # First occurrence - store as single value
                        self.graph[author][entity][meta_key] = value
                    else:
                        # Multiple values - convert to list or append
                        existing = self.graph[author][entity][meta_key]
                        if isinstance(existing, list):
                            # Already a list - append if not duplicate
                            if value not in existing:
                                existing.append(value)
                        else:
                            # Convert to list if values differ
                            if existing != value:
                                self.graph[author][entity][meta_key] = [existing, value]
        else:
            # Create new edge
            edge_attrs = {
                'weight': 1,
                'entity_type': entity_type,
                'avg_score': score
            }

            if post_id:
                edge_attrs['source_posts'] = [post_id]

            if timestamp:
                edge_attrs['first_mention'] = timestamp
                edge_attrs['last_mention'] = timestamp

            # Add edge metadata if provided
            if edge_metadata:
                for key, value in edge_metadata.items():
                    if key not in edge_attrs:  # Don't override core attributes
                        edge_attrs[f'meta_{key}'] = value

            self.graph.add_edge(author, entity, **edge_attrs)
            self.stats['edges_created'] += 1

    def _add_author_mention_edge(
        self,
        source_author: str,
        target_author: str,
        post_id: Optional[str],
        timestamp: Optional[str]
    ):
        """Add or update edge from one author to another."""

        # Don't create self-loops
        if source_author == target_author:
            return

        # Ensure target author node exists
        if not self.graph.has_node(target_author):
            self.graph.add_node(
                target_author,
                node_type='author',
                label=target_author,
                mention_count=0,
                post_count=0
            )

        # Increment mention count for target author
        self.graph.nodes[target_author]['mention_count'] += 1

        # Add or update edge
        if self.graph.has_edge(source_author, target_author):
            # Edge exists, increment weight
            self.graph[source_author][target_author]['weight'] += 1

            if post_id:
                if 'source_posts' not in self.graph[source_author][target_author]:
                    self.graph[source_author][target_author]['source_posts'] = []
                self.graph[source_author][target_author]['source_posts'].append(post_id)

            if timestamp:
                self.graph[source_author][target_author]['last_mention'] = timestamp
        else:
            # Create new edge
            edge_attrs = {
                'weight': 1,
                'entity_type': 'AUTHOR'
            }

            if post_id:
                edge_attrs['source_posts'] = [post_id]

            if timestamp:
                edge_attrs['first_mention'] = timestamp
                edge_attrs['last_mention'] = timestamp

            self.graph.add_edge(source_author, target_author, **edge_attrs)
            self.stats['edges_created'] += 1

    def finalize_network(self) -> nx.DiGraph:
        """
        Finalize the network and return the graph.

        Returns:
            NetworkX DiGraph object
        """
        logger.info(f"Network finalized: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        return self.graph

    def get_graph(self) -> nx.DiGraph:
        """Get the current network graph."""
        return self.graph

    def get_statistics(self) -> Dict:
        """
        Calculate and return network statistics.

        Returns:
            Dictionary with network statistics
        """
        if len(self.graph.nodes) == 0:
            return {
                'total_nodes': 0,
                'total_edges': 0,
                'density': 0.0,
                'posts_processed': self.stats['posts_processed']
            }

        # Count nodes by type
        node_types = Counter()
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('node_type', 'unknown')
            node_types[node_type] += 1

        # Count edges by type
        edge_types = Counter()
        total_weight = 0
        for u, v, attrs in self.graph.edges(data=True):
            edge_type = attrs.get('entity_type', 'unknown')
            edge_types[edge_type] += 1
            total_weight += attrs.get('weight', 1)

        # Calculate density
        n_nodes = len(self.graph.nodes)
        n_edges = len(self.graph.edges)
        max_edges = n_nodes * (n_nodes - 1)  # Directed graph
        density = n_edges / max_edges if max_edges > 0 else 0.0

        # Get top mentioned entities
        entities_by_mentions = [
            (node, attrs.get('mention_count', 0), attrs.get('node_type', 'unknown'))
            for node, attrs in self.graph.nodes(data=True)
            if attrs.get('node_type') != 'author'
        ]
        entities_by_mentions.sort(key=lambda x: x[1], reverse=True)
        top_entities = [
            {'entity': e[0], 'mentions': e[1], 'type': e[2]}
            for e in entities_by_mentions[:10]
        ]

        # Calculate degree statistics
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())

        avg_in_degree = sum(in_degrees.values()) / n_nodes if n_nodes > 0 else 0
        avg_out_degree = sum(out_degrees.values()) / n_nodes if n_nodes > 0 else 0

        # Find connected components
        try:
            n_components = nx.number_weakly_connected_components(self.graph)
            largest_component_size = len(max(
                nx.weakly_connected_components(self.graph),
                key=len
            )) if n_nodes > 0 else 0
        except:
            n_components = 0
            largest_component_size = 0

        return {
            # Basic counts
            'total_nodes': n_nodes,
            'total_edges': n_edges,
            'density': density,

            # Node type counts
            'authors': node_types.get('author', 0),
            'persons': node_types.get('person', 0),
            'locations': node_types.get('location', 0),
            'organizations': node_types.get('organization', 0),

            # Edge type counts
            'person_mentions': edge_types.get('PER', 0) + edge_types.get('PERSON', 0),
            'location_mentions': edge_types.get('LOC', 0) + edge_types.get('LOCATION', 0),
            'organization_mentions': edge_types.get('ORG', 0) + edge_types.get('ORGANIZATION', 0),
            'author_mentions': edge_types.get('AUTHOR', 0),

            # Weight statistics
            'total_mentions': total_weight,
            'avg_mentions_per_edge': total_weight / n_edges if n_edges > 0 else 0,

            # Degree statistics
            'avg_in_degree': avg_in_degree,
            'avg_out_degree': avg_out_degree,

            # Component statistics
            'connected_components': n_components,
            'largest_component_size': largest_component_size,

            # Top entities
            'top_entities': top_entities,

            # Processing statistics
            'posts_processed': self.stats['posts_processed'],
            'entities_added': self.stats['entities_added'],
            'edges_created': self.stats['edges_created'],
            'author_self_mentions': self.stats['author_mentions']
        }

    def get_node_info(self, node_id: str) -> Optional[Dict]:
        """
        Get information about a specific node.

        Args:
            node_id: Node identifier

        Returns:
            Dictionary with node information or None if not found
        """
        if not self.graph.has_node(node_id):
            return None

        attrs = self.graph.nodes[node_id]
        in_degree = self.graph.in_degree(node_id)
        out_degree = self.graph.out_degree(node_id)

        return {
            'id': node_id,
            'type': attrs.get('node_type', 'unknown'),
            'label': attrs.get('label', node_id),
            'mention_count': attrs.get('mention_count', 0),
            'post_count': attrs.get('post_count', 0),
            'in_degree': in_degree,
            'out_degree': out_degree,
            'total_degree': in_degree + out_degree
        }

    def get_top_authors(self, n: int = 10) -> List[Dict]:
        """
        Get top N authors by post count.

        Args:
            n: Number of authors to return

        Returns:
            List of author dictionaries
        """
        authors = [
            {
                'author': node,
                'posts': attrs.get('post_count', 0),
                'mentions': attrs.get('mention_count', 0),
                'out_degree': self.graph.out_degree(node)
            }
            for node, attrs in self.graph.nodes(data=True)
            if attrs.get('node_type') == 'author'
        ]

        authors.sort(key=lambda x: x['posts'], reverse=True)
        return authors[:n]

    def get_edge_info(self, source: str, target: str) -> Optional[Dict]:
        """
        Get information about a specific edge.

        Args:
            source: Source node
            target: Target node

        Returns:
            Dictionary with edge information or None if not found
        """
        if not self.graph.has_edge(source, target):
            return None

        attrs = self.graph[source][target]
        return {
            'source': source,
            'target': target,
            'weight': attrs.get('weight', 1),
            'entity_type': attrs.get('entity_type', 'unknown'),
            'source_posts': attrs.get('source_posts', []),
            'first_mention': attrs.get('first_mention'),
            'last_mention': attrs.get('last_mention')
        }

    def reset(self):
        """Clear the network and reset statistics."""
        self.graph.clear()
        self.authors.clear()
        self.stats = {
            'posts_processed': 0,
            'entities_added': 0,
            'edges_created': 0,
            'author_mentions': 0
        }
        if self.entity_resolver:
            self.entity_resolver.reset()

        logger.info("Network builder reset")


# Example usage
if __name__ == "__main__":
    # Create network builder
    builder = NetworkBuilder()

    # Simulate adding posts with entities
    posts = [
        {
            'author': '@user1',
            'entities': [
                {'text': 'John Smith', 'type': 'PER', 'score': 0.95},
                {'text': 'Microsoft', 'type': 'ORG', 'score': 0.92},
                {'text': 'Copenhagen', 'type': 'LOC', 'score': 0.89}
            ],
            'post_id': 'post_1'
        },
        {
            'author': '@user2',
            'entities': [
                {'text': 'john smith', 'type': 'PER', 'score': 0.93},  # Same as above
                {'text': 'Paris', 'type': 'LOC', 'score': 0.91}
            ],
            'post_id': 'post_2'
        },
        {
            'author': '@user1',
            'entities': [
                {'text': 'Microsoft', 'type': 'ORG', 'score': 0.94},  # Duplicate
                {'text': 'Google', 'type': 'ORG', 'score': 0.90}
            ],
            'post_id': 'post_3'
        }
    ]

    # Add posts to network
    for post in posts:
        builder.add_post(
            author=post['author'],
            entities=post['entities'],
            post_id=post['post_id']
        )

    # Get statistics
    stats = builder.get_statistics()

    print("Network Statistics:")
    print(f"  Nodes: {stats['total_nodes']}")
    print(f"  Edges: {stats['total_edges']}")
    print(f"  Authors: {stats['authors']}")
    print(f"  Persons: {stats['persons']}")
    print(f"  Locations: {stats['locations']}")
    print(f"  Organizations: {stats['organizations']}")
    print(f"  Density: {stats['density']:.4f}")

    print("\nTop Entities:")
    for entity in stats['top_entities']:
        print(f"  {entity['entity']} ({entity['type']}): {entity['mentions']} mentions")

    print("\nTop Authors:")
    for author in builder.get_top_authors(5):
        print(f"  {author['author']}: {author['posts']} posts, {author['out_degree']} connections")
