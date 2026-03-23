import ast
from pathlib import Path
from pyvis.network import Network
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path('/content/drive/MyDrive/hybridvision')

class ArchitectureVisitor(ast.NodeVisitor):
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.current_class = None

    def visit_ClassDef(self, node):
        class_name = node.name
        logging.info(f"  -> Found class definition: {class_name}")
        self.nodes[class_name] = {'type': 'class', 'file': None}
        self.current_class = class_name
        
        for base in node.bases:
            if isinstance(base, ast.Name):
                logging.info(f"     - Found inheritance: {class_name} inherits from {base.id}")
                self.edges.append({'from': class_name, 'to': base.id, 'type': 'inherits'})
        
        self.generic_visit(node)
        self.current_class = None

    def visit_Assign(self, node):
        if (self.current_class and 
                isinstance(node.targets[0], ast.Attribute) and
                isinstance(node.targets[0].value, ast.Name) and
                node.targets[0].value.id == 'self' and
                isinstance(node.value, ast.Call) and
                isinstance(node.value.func, ast.Name)):
            
            component_class = node.value.func.id
            logging.info(f"     - Found usage: {self.current_class} uses {component_class}")
            self.edges.append({'from': self.current_class, 'to': component_class, 'type': 'uses'})
        
        self.generic_visit(node)

def analyze_file(filepath, visitor):
    logging.info(f"Analyzing file: {filepath.relative_to(PROJECT_ROOT)}")
    try:
        source = filepath.read_text(encoding='utf-8')
        tree = ast.parse(source)
        visitor.visit(tree)
        
        # Associa o nome do arquivo a todas as classes encontradas nele
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in visitor.nodes:
                visitor.nodes[node.name]['file'] = str(filepath.relative_to(PROJECT_ROOT))

    except Exception as e:
        logging.error(f"Could not parse {filepath}: {e}")

def create_graph(nodes, edges):
    net = Network(height="800px", width="100%", notebook=True, cdn_resources='in_line', directed=True)
    net.set_options("""
    var options = {
      "nodes": { "font": { "size": 16 } },
      "edges": { "font": { "size": 12, "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } } } },
      "physics": { "solver": "barnesHut", "barnesHut": { "gravitationalConstant": -30000, "centralGravity": 0.1, "springLength": 250 }, "minVelocity": 0.75 }
    }
    """)

    for node_name, properties in nodes.items():
        path_str = properties.get('file', '')
        color = '#6c757d'
        if 'src' in path_str: color = '#007bff'
        elif 'experiments' in path_str: color = '#28a745'
        elif 'scripts' in path_str: color = '#fd7e14'
        net.add_node(node_name, label=node_name, title=f"File: {path_str}", shape='box', color=color)

    for edge in edges:
        if edge['from'] in nodes and edge['to'] in nodes:
            if edge['type'] == 'inherits':
                net.add_edge(edge['from'], edge['to'], label='inherits', color='#17a2b8')
            elif edge['type'] == 'uses':
                net.add_edge(edge['from'], edge['to'], label='uses', dashes=True, color='#6c757d')
    return net

def main():
    visitor = ArchitectureVisitor()
    paths_to_analyze = [PROJECT_ROOT / 'src', PROJECT_ROOT / 'experiments', PROJECT_ROOT / 'scripts']
    
    for path in paths_to_analyze:
        if path.is_dir():
            logging.info(f"--- Scanning directory: {path} ---")
            for py_file in sorted(path.rglob('*.py')):
                if py_file.name != '__init__.py':
                    analyze_file(py_file, visitor)
    
    dashboard_path = PROJECT_ROOT / 'dashboard.py'
    if dashboard_path.exists():
        analyze_file(dashboard_path, visitor)
    
    logging.info("--- Analysis Summary ---")
    logging.info(f"Total nodes (classes) found: {len(visitor.nodes)}")
    logging.info(f"Total edges (relationships) found: {len(visitor.edges)}")
    if not visitor.nodes:
        logging.error("No classes were found. The resulting graph will be empty.")
        return 

    network = create_graph(visitor.nodes, visitor.edges)
    output_dir = PROJECT_ROOT / 'docs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / "architecture.html"
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(network.html)
        logging.info(f"Interactive architecture graph saved to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to write HTML file: {e}")

if __name__ == '__main__':
    main()