# Migration Guide: Front-End Force Atlas 2 Visualization

## Overview

This guide explains how to migrate from Python `fa2` package to a JavaScript-based Force Atlas 2 visualization similar to [4cat's implementation](https://github.com/digitalmethodsinitiative/4cat).

## Current Architecture

**Current (Python-based):**
```
Data → NetworkX Graph → fa2 (Python) → Plotly → Streamlit
```

**Proposed (Browser-based):**
```
Data → NetworkX Graph → JSON Export → Sigma.js + FA2 (JS) → Streamlit
```

## Benefits of Front-End Visualization

✅ **Performance**
- Layout computation happens in browser (parallel to other processing)
- No Python FA2 dependency needed
- Faster for interactive adjustments

✅ **Interactivity**
- Real-time layout parameter adjustments
- Dynamic node filtering
- Better zoom/pan controls
- Live layout animation

✅ **User Experience**
- Smoother interactions
- No page reloads for layout changes
- Professional network visualization

✅ **Compatibility**
- Works without fa2 installation issues
- Cross-platform (runs in any browser)
- Mobile-friendly

---

## Implementation Options

### Option 1: Sigma.js + Graphology (Recommended - Same as 4cat)

**Pros:**
- Same implementation as 4cat
- Optimized for large networks
- Built-in Force Atlas 2
- WebGL rendering (very fast)

**Cons:**
- Need to integrate JavaScript into Streamlit
- More complex initial setup

**Libraries:**
```html
<script src="https://cdn.jsdelivr.net/npm/graphology@0.25.0/dist/graphology.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/sigma@2.4.0/sigma.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/graphology-layout-forceatlas2@0.10.0/index.min.js"></script>
```

### Option 2: D3.js Force Simulation

**Pros:**
- Widely used and documented
- Flexible and customizable
- Good Streamlit integration examples

**Cons:**
- Not exactly Force Atlas 2 (similar but different algorithm)
- Slower for very large networks (>1000 nodes)

**Libraries:**
```html
<script src="https://d3js.org/d3.v7.min.js"></script>
```

### Option 3: vis-network

**Pros:**
- Very easy to use
- Good documentation
- Physics simulation similar to FA2

**Cons:**
- Different algorithm
- Less customization

---

## Recommended Implementation (Sigma.js)

### Step 1: Update `visualizer.py`

Remove FA2 computation, add JSON export:

```python
# src/utils/visualizer.py

class NetworkVisualizer:
    """Export network data for front-end visualization."""

    def export_for_sigma(self, graph: nx.DiGraph) -> dict:
        """
        Export graph data in Sigma.js format.

        Returns:
            Dictionary with nodes and edges for Sigma.js
        """
        nodes = []
        edges = []

        for node_id, data in graph.nodes(data=True):
            nodes.append({
                'key': str(node_id),
                'label': data.get('label', str(node_id)),
                'size': 10 + data.get('mention_count', 0) * 2,
                'color': self._get_node_color(data.get('node_type', 'unknown')),
                'type': data.get('node_type', 'unknown'),
                'mention_count': data.get('mention_count', 0),
                'post_count': data.get('post_count', 0)
            })

        for i, (u, v, data) in enumerate(graph.edges(data=True)):
            edges.append({
                'key': f'edge_{i}',
                'source': str(u),
                'target': str(v),
                'weight': data.get('weight', 1),
                'type': data.get('entity_type', 'default')
            })

        return {
            'nodes': nodes,
            'edges': edges
        }

    def _get_node_color(self, node_type: str) -> str:
        """Get color for node type."""
        return {
            'author': '#1f77b4',
            'person': '#ff7f0e',
            'location': '#2ca02c',
            'organization': '#d62728'
        }.get(node_type.lower(), '#888888')
```

### Step 2: Create HTML Template

Create `src/cli/templates/sigma_viewer.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Network Visualization</title>
    <style>
        #container {
            width: 100%;
            height: 800px;
            position: relative;
        }
        #controls {
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            z-index: 1000;
        }
        #controls button {
            display: block;
            width: 100%;
            margin: 5px 0;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            cursor: pointer;
        }
        #controls button:hover {
            background: #f0f0f0;
        }
        #controls label {
            display: block;
            margin-top: 10px;
            font-size: 12px;
        }
        #controls input[type="range"] {
            width: 100%;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/graphology@0.25.0/dist/graphology.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/sigma@2.4.0/sigma.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/graphology-layout-forceatlas2@0.10.0/index.min.js"></script>
</head>
<body>
    <div id="container">
        <div id="controls">
            <h3 style="margin-top: 0;">Layout Controls</h3>
            <button id="start-layout">▶️ Start Layout</button>
            <button id="stop-layout">⏸️ Stop Layout</button>
            <button id="reset-layout">🔄 Reset</button>

            <label>
                Gravity: <span id="gravity-value">1.0</span>
                <input type="range" id="gravity" min="0" max="5" step="0.1" value="1.0">
            </label>

            <label>
                Scaling: <span id="scaling-value">10</span>
                <input type="range" id="scaling" min="1" max="50" step="1" value="10">
            </label>

            <label>
                Edge Weight: <span id="edge-weight-value">1.0</span>
                <input type="range" id="edge-weight" min="0" max="2" step="0.1" value="1.0">
            </label>

            <label>
                <input type="checkbox" id="barnes-hut" checked> Barnes-Hut Optimization
            </label>
        </div>
    </div>

    <script>
        // Graph data will be injected here
        const graphData = {{GRAPH_DATA}};

        // Create graph
        const graph = new graphology.Graph();

        // Add nodes and edges
        graphData.nodes.forEach(node => {
            graph.addNode(node.key, node);
        });

        graphData.edges.forEach(edge => {
            graph.addEdge(edge.source, edge.target, edge);
        });

        // Create Sigma renderer
        const container = document.getElementById('container');
        const renderer = new Sigma(graph, container, {
            renderEdgeLabels: false,
            defaultNodeColor: '#999',
            defaultEdgeColor: '#ccc',
            labelSize: 14,
            labelWeight: 'bold'
        });

        // Force Atlas 2 settings
        let layoutRunning = false;
        let sensibleSettings = {
            adjustSizes: false,
            barnesHutOptimize: true,
            barnesHutTheta: 1.2,
            edgeWeightInfluence: 1.0,
            gravity: 1.0,
            linLogMode: false,
            outboundAttractionDistribution: true,
            scalingRatio: 10,
            slowDown: 1,
            strongGravityMode: false
        };

        // Layout controls
        document.getElementById('start-layout').addEventListener('click', () => {
            if (!layoutRunning) {
                layoutRunning = true;
                graphologyLibrary.layoutForceAtlas2.assign(graph, {
                    iterations: 50,
                    settings: sensibleSettings
                });
                renderer.refresh();
            }
        });

        document.getElementById('stop-layout').addEventListener('click', () => {
            layoutRunning = false;
        });

        document.getElementById('reset-layout').addEventListener('click', () => {
            graph.forEachNode((node) => {
                graph.setNodeAttribute(node, 'x', Math.random());
                graph.setNodeAttribute(node, 'y', Math.random());
            });
            renderer.refresh();
        });

        // Update settings
        document.getElementById('gravity').addEventListener('input', (e) => {
            sensibleSettings.gravity = parseFloat(e.target.value);
            document.getElementById('gravity-value').textContent = e.target.value;
        });

        document.getElementById('scaling').addEventListener('input', (e) => {
            sensibleSettings.scalingRatio = parseFloat(e.target.value);
            document.getElementById('scaling-value').textContent = e.target.value;
        });

        document.getElementById('edge-weight').addEventListener('input', (e) => {
            sensibleSettings.edgeWeightInfluence = parseFloat(e.target.value);
            document.getElementById('edge-weight-value').textContent = e.target.value;
        });

        document.getElementById('barnes-hut').addEventListener('change', (e) => {
            sensibleSettings.barnesHutOptimize = e.target.checked;
        });

        // Initial layout
        graphologyLibrary.layoutForceAtlas2.assign(graph, {
            iterations: 100,
            settings: sensibleSettings
        });
        renderer.refresh();
    </script>
</body>
</html>
```

### Step 3: Update Streamlit App

Modify `src/cli/app.py` to use the new visualization:

```python
import streamlit.components.v1 as components
import json

def display_network_visualization(graph, stats):
    """Display interactive Sigma.js network visualization."""

    # Export graph data
    viz = NetworkVisualizer()
    graph_data = viz.export_for_sigma(graph)

    # Load HTML template
    template_path = Path(__file__).parent / 'templates' / 'sigma_viewer.html'
    with open(template_path, 'r') as f:
        html_template = f.read()

    # Inject graph data
    html_content = html_template.replace(
        '{{GRAPH_DATA}}',
        json.dumps(graph_data)
    )

    # Display in Streamlit
    components.html(html_content, height=850, scrolling=False)
```

### Step 4: Remove fa2 Dependency

Update `requirements.txt`:

```diff
- fa2>=0.3.5
+ # fa2 removed - using front-end visualization
```

Update `setup.py`:

```diff
- 'fa2>=0.3.5',
+ # fa2 removed - using front-end visualization
```

---

## Migration Checklist

### Phase 1: Preparation
- [ ] Review current visualizer.py usage
- [ ] Backup current implementation
- [ ] Create templates directory: `mkdir -p src/cli/templates`
- [ ] Test Sigma.js locally with sample data

### Phase 2: Implementation
- [ ] Create new `export_for_sigma()` method in visualizer.py
- [ ] Create sigma_viewer.html template
- [ ] Update app.py to use new visualization
- [ ] Test with small network (<100 nodes)
- [ ] Test with medium network (100-1000 nodes)
- [ ] Test with large network (>1000 nodes)

### Phase 3: Cleanup
- [ ] Remove fa2 import from visualizer.py
- [ ] Remove fa2 from requirements.txt
- [ ] Remove fa2 from setup.py
- [ ] Update documentation
- [ ] Update README.md

### Phase 4: Enhancement (Optional)
- [ ] Add node filtering UI
- [ ] Add search functionality
- [ ] Add export current view as image
- [ ] Add color scheme selector
- [ ] Add layout presets

---

## Code Changes Summary

### Files to Modify:
1. `src/utils/visualizer.py` - Add JSON export, remove FA2 computation
2. `src/cli/app.py` - Use HTML component instead of Plotly
3. `requirements.txt` - Remove fa2
4. `setup.py` - Remove fa2

### Files to Create:
1. `src/cli/templates/sigma_viewer.html` - Sigma.js visualization template
2. `src/cli/templates/d3_viewer.html` - Alternative D3.js version (optional)

### Files to Update:
1. `README.md` - Update visualization section
2. `ERROR_HANDLING_GUIDE.md` - Remove fa2 warning
3. `CHANGELOG.md` - Document the change

---

## Testing

```bash
# Install dependencies (fa2 removed)
pip install -e .

# Run Streamlit app
streamlit run src/cli/app.py

# Test with sample data
python examples/test_pipeline.py
```

---

## Rollback Plan

If issues occur, restore previous version:

```bash
git checkout -- src/utils/visualizer.py
git checkout -- src/cli/app.py
pip install fa2
```

---

## Performance Comparison

| Metric | Python FA2 | Browser FA2 (Sigma.js) |
|--------|-----------|------------------------|
| 100 nodes | ~1-2s | < 0.5s (client-side) |
| 1000 nodes | ~5-10s | ~1-2s (client-side) |
| 5000 nodes | ~30-60s | ~5-10s (client-side) |
| Interactivity | Static | Real-time |
| Memory | Server RAM | Client RAM |

---

## Troubleshooting

**Issue: Visualization not showing**
- Check browser console for errors
- Verify graph data JSON is valid
- Ensure CDN scripts are loading

**Issue: Layout looks wrong**
- Adjust gravity/scaling parameters
- Increase iterations
- Try Barnes-Hut optimization

**Issue: Slow performance**
- Enable Barnes-Hut optimization
- Reduce number of layout iterations
- Filter nodes before visualization

---

## Resources

- [Sigma.js Documentation](https://www.sigmajs.org/)
- [Graphology Documentation](https://graphology.github.io/)
- [Force Atlas 2 Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679)
- [4cat Sigma Implementation](https://github.com/digitalmethodsinitiative/4cat/blob/master/webtool/static/js/sigma_network.js)
- [Streamlit Components](https://docs.streamlit.io/library/components)

---

**Estimated Time**: 2-4 hours
**Difficulty**: Medium
**Risk**: Low (can rollback easily)
**Benefit**: High (better UX, no fa2 dependency issues)
