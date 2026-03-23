import ast
from pathlib import Path
import logging
import os
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path('G:/Meu Drive/hybridvision')

def analyze_source_code(files_to_scan):
    nodes = {}
    edges = []

    for py_file in files_to_scan:
        try:
            source = py_file.read_text(encoding='utf-8')
            tree = ast.parse(source)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    
                    if class_name == 'ArchitectureVisitor':
                        continue

                    if class_name not in nodes:
                        nodes[class_name] = {
                            'id': class_name, 'label': class_name, 'group': 'class',
                            'shape': 'box', 'file': str(py_file.relative_to(PROJECT_ROOT))
                        }

                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            edges.append({'source': class_name, 'target': base.id, 'label': 'inherits'})
                    
                    for method_node in node.body:
                        if isinstance(method_node, ast.FunctionDef):
                            method_name = method_node.name
                            
                            if method_name == '__init__':
                                continue
                                
                            method_id = f"{class_name}::{method_name}"
                            method_group = 'private_method' if method_name.startswith('_') else 'public_method'
                            
                            if method_id not in nodes:
                                nodes[method_id] = {
                                    'id': method_id, 'label': f"{method_name}()", 'group': method_group,
                                    'shape': 'ellipse', 'title': f"Method of {class_name}"
                                }
                                edges.append({'source': class_name, 'target': method_id, 'label': 'contains'})

                    for sub_node in ast.walk(node):
                        if (isinstance(sub_node, ast.Assign) and
                                isinstance(sub_node.targets[0], ast.Attribute) and
                                isinstance(sub_node.targets[0].value, ast.Name) and
                                sub_node.targets[0].value.id == 'self' and
                                isinstance(sub_node.value, ast.Call) and
                                isinstance(sub_node.value.func, ast.Name)):
                            component_class = sub_node.value.func.id
                            edges.append({'source': class_name, 'target': component_class, 'label': 'uses'})

        except Exception as e:
            logging.error(f"Could not parse {py_file} during analysis: {e}")
            
    return list(nodes.values()), edges


def main():
    all_py_files = set()
    paths_to_scan = [PROJECT_ROOT]
    for path in paths_to_scan:
        if path.is_dir():
            all_py_files.update(py_file for py_file in path.rglob('*.py') if py_file.name != '__init__.py')

    logging.info(f"--- Starting Architecture Analysis: Found {len(all_py_files)} Python files ---")
    
    nodes, edges = analyze_source_code(sorted(list(all_py_files)))

    manual_nodes = [
        {'id': 'run.py', 'label': 'run.py', 'group': 'script', 'shape': 'box', 'file': 'run.py'},
        {'id': 'tuner.py', 'label': 'tuner.py', 'group': 'script', 'shape': 'box', 'file': 'experiments/tuner.py'},
        {'id': 'create_clusterer', 'label': 'create_clusterer()', 'group': 'function', 'shape': 'diamond', 'file': 'src/clusterers/utils_clusterer.py'},
        {'id': 'get_normalizer', 'label': 'get_normalizer()', 'group': 'function', 'shape': 'diamond', 'file': 'src/normalizers/feature_normalizer.py'}
    ]
    nodes.extend(manual_nodes)

    manual_edges = [
        {'source': 'Pipeline', 'target': 'create_clusterer', 'label': 'uses'},
        {'source': 'create_clusterer', 'target': 'KMeansClusterer', 'label': 'calls'},
        {'source': 'create_clusterer', 'target': 'GraphClusterer', 'label': 'calls'},
        {'source': 'Pipeline', 'target': 'get_normalizer', 'label': 'uses'},
        {'source': 'get_normalizer', 'target': 'FeatureNormalizer', 'label': 'calls'},
        {'source': 'Pipeline', 'target': 'KValueOptimizer', 'label': 'uses'},
        {'source': 'run.py', 'target': 'PerformanceAnalyzer', 'label': 'uses'},
        {'source': 'run.py', 'target': 'Pipeline', 'label': 'uses'},
        {'source': 'tuner.py', 'target': 'Pipeline', 'label': 'uses'},
    ]
    edges.extend(manual_edges)
    
    output_path = PROJECT_ROOT / 'docs' / "architecture_data.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'nodes': nodes, 'edges': edges}, f, indent=2)
        
    logging.info(f"Analysis complete. Found {len(nodes)} nodes and {len(edges)} edges.")
    logging.info(f"Architecture data saved to: {output_path}")

if __name__ == '__main__':
    main()