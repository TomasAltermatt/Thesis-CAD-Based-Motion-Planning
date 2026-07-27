import networkx as nx
from pathlib import Path
import IM_Generation.functions as imf
import time
import os
import pickle

def build_geometric_and_or_graph(matrices, assembly_manifest):
    """
    Takes the 6 directional Interference Matrices and builds a Directed 
    AND/OR Multigraph of all valid geometric disassembly sequences.
    Generates parallel arcs for different holding arms and directions.
    """
    # 1. Guarantee part names align exactly with matrix indices
    sorted_parts = sorted(assembly_manifest.items(), key=lambda x: x[1]["matrix_idx"])
    part_names = [name for name, data in sorted_parts]
    N = len(part_names)
    
    # 2. Initialize the Directed Multigraph (Allows parallel edges)
    G = nx.MultiDiGraph()
    
    # 3. The Root Node: All parts present (tuple of 1s)
    root_state = tuple([1] * N)
    G.add_node(root_state)
    
    # Queue for Breadth-First Search
    queue = [root_state]
    visited = set([root_state])
    
    # Explicitly define the available robotic arms for the parallel edges
    arms = ["Left", "Right"]
    
    print(f"\n[STARTING] Building AND/OR Multigraph for {N} parts...")
    
    while queue:
        current_state = queue.pop(0)
        
        # If the state is (0, 0, 0...), the assembly is fully disassembled. Stop here.
        if sum(current_state) == 0:
            continue
            
        # Get the indices of the parts that are still physically present in this state
        active_indices = [i for i, val in enumerate(current_state) if val == 1]
        
        # Test each active part to see if it can be extracted
        for i in active_indices:
            
            # Check all 6 extraction directions
            for direction, matrix in matrices.items():
                
                # Assume it's free until we find a collision
                collision_free = True
                
                # Check against all OTHER parts currently in the sub-assembly
                for j in active_indices:
                    if i != j:
                        # If there is a '2', part 'j' blocks part 'i' in this direction
                        if matrix[i, j] == 2:
                            collision_free = False
                            break 
                            
                # If no collisions were found, this is a valid extraction path!
                if collision_free:
                    # Create the new child state by removing part 'i'
                    new_state_list = list(current_state)
                    new_state_list[i] = 0
                    new_state = tuple(new_state_list)
                    
                    # If we haven't seen this specific sub-assembly state before, add it to the graph
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append(new_state)
                        G.add_node(new_state)
                        
                    # MULTIGRAPH LOGIC: Generate parallel arcs for each holding arm
                    for holding_arm in arms:
                        G.add_edge(
                            current_state, 
                            new_state, 
                            removed_part=part_names[i], 
                            removed_idx=i,
                            direction=direction,
                            holding_arm=holding_arm
                        )
                        
    print(f"--- Graph Complete! Generated {G.number_of_nodes()} Nodes and {G.number_of_edges()} Edges ---")
    return G

def print_geometric_assembly_sequences(G, N):
    """
    Prints all unique geometric assembly sequences from a MultiDiGraph,
    ignoring the parallel robotic arcs (direction/arms).
    """
    start_node = tuple([1] * N)  # Fully assembled
    end_node = tuple([0] * N)    # Fully disassembled
    
    # THE FIX: Cast to a standard DiGraph purely for the pathfinding.
    # This instantly collapses all parallel robotic edges into a single geometric edge,
    # preventing the combinatorial explosion of duplicate paths.
    G_simple = nx.DiGraph(G)
    
    all_paths = list(nx.all_simple_paths(G_simple, source=start_node, target=end_node))
    
    print(f"\nFound {len(all_paths)} unique geometric assembly sequences.")
    
    for path_idx, path in enumerate(all_paths):
        disassembly_sequence = []
        
        # Walk through the nodes in this specific sequence
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            
            # We still pull the part name from the original Multigraph 'G'
            # because G_simple might strip out the edge dictionary attributes
            first_edge_key = list(G[u][v].keys())[0]
            removed_part = G[u][v][first_edge_key]['removed_part']
            
            disassembly_sequence.append(removed_part)
            
        # Reverse the disassembly list to get the Assembly sequence
        assembly_sequence = list(reversed(disassembly_sequence))
        
        # Print the clean sequence
        print(f"Sequence {path_idx + 1}: {' -> '.join(assembly_sequence)}")


def build_fabrica_precedence_graph(matrices, assembly_manifest):
    """
    Builds a strict Fabrica-style Directed Precedence Graph using Interference Matrices.
    Nodes = Parts. Edges = Assembly dependencies (Part A must be assembled before Part B).
    """
    # 1. Map part names to their matrix indices
    sorted_parts = sorted(assembly_manifest.items(), key=lambda x: x[1]["matrix_idx"])
    part_names = [name for name, data in sorted_parts]
    N = len(part_names)
    
    G = nx.DiGraph()
    
    # Trackers for the tier loop
    active_indices = set(range(N)) # Starts with all parts in the sub-assembly
    removed_indices = set()        # Starts empty
    tiers = []                     # Records the disassembly order
    
    print(f"\n[STARTING] Building Precedence Graph for {N} parts...")
    
    while len(active_indices) > 1:
        current_tier = {}
        
        # --- PHASE 1: Find the unblocked tier ---
        for i in active_indices:
            valid_dir = None
            
            for direction, matrix in matrices.items():
                is_free = True
                # Check interference ONLY against parts still in the active sub-assembly
                for j in active_indices:
                    if i != j and matrix[i, j] == 2:
                        is_free = False
                        break 
                
                if is_free:
                    valid_dir = direction
                    break # Stop at the first valid direction for this part
            
            if valid_dir:
                current_tier[i] = valid_dir
                
        # Deadlock Failsafe
        if not current_tier:
            raise ValueError("Deadlock detected! No remaining parts can be extracted.")
            
        # --- PHASE 2: Add Nodes and Edges ---
        for i, direction in current_tier.items():
            part_name = part_names[i]
            action_vector = assembly_manifest[part_name]["extraction_vectors"][direction]
            
            # -> [YOUR SIMULATOR HOOK GOES HERE] <-
            # Here is where you will run your lightweight path simulator 
            # for 'part_name' strictly in 'direction' to generate the 6D path array.
            dummy_path = None 
            
            # Add the part to the graph
            G.add_node(
                part_name, 
                action=action_vector, 
                path=dummy_path, 
                parts_before=[], 
                parts_after=[]
            )
            
            # Draw dependencies: Check extraction path against ALREADY REMOVED parts
            # If extracting 'i' collides with a removed part 'k', then 'k' blocked 'i'.
            # Therefore, in forward-assembly, 'i' must be assembled BEFORE 'k'.
            for k in removed_indices:
                if matrices[direction][i, k] == 2:
                    G.add_edge(part_name, part_names[k])
                    
        # --- PHASE 3: Update Trackers ---
        active_indices -= set(current_tier.keys())
        removed_indices |= set(current_tier.keys())
        tiers.append(list(current_tier.keys()))
        
    # --- PHASE 4: Handle the Base Part ---
    if len(active_indices) == 1:
        base_idx = active_indices.pop()
        base_name = part_names[base_idx]
        
        G.add_node(
            base_name, 
            action=None, 
            path=None, 
            parts_before=[], 
            parts_after=[]
        )
        
        # The base part must strictly be assembled before anything in the last removed tier
        if tiers:
            for k in tiers[-1]:
                G.add_edge(base_name, part_names[k])
                
    print(f"--- Graph Complete! Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()} ---")
    return G

def draw_graph(G, save_path=None):
    from networkx.drawing.nx_pydot import graphviz_layout
    import matplotlib.pyplot as plt
    import networkx as nx
    
    # 1. Make the canvas much larger (Width, Height in inches)
    plt.figure(figsize=(16, 10))
    
    pos = graphviz_layout(G, prog='dot')
    
    # 2. Tweak the drawing parameters for readability
    nx.draw_networkx(
        G, 
        pos, 
        arrows=True, 
        with_labels=True,
        node_size=1000,         # Make nodes larger to fit text
        node_color="skyblue",   # Lighter color so black text is readable
        font_size=9,            # Shrink the text slightly
        font_weight="bold",
        edge_color="gray"       # Push edges to the background
    )
    
    # 3. Trim giant white borders
    plt.tight_layout()
    
    if save_path is None:
        plt.show()
    else:
        # 4. Save with high resolution (dpi=300) so you can zoom in
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() # Close the figure to free up memory


def save_graph(G, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    graph_path = os.path.join(log_dir, 'precedence.pkl')
    image_path = os.path.join(log_dir, 'precedence.png')
    with open(graph_path, 'wb') as fp:
        pickle.dump(G, fp)
    draw_graph(G, save_path=image_path)