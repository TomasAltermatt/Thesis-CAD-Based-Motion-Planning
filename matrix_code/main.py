import networkx as nx
from pathlib import Path
import IM_Generation.functions as imf
from Graph_Sequencing.functions import *
import time

if __name__ == "__main__":
    # Change this whenever you test a new assembly
    input_folder = 'STLs/beam'
    
    # Extract the assembly name dynamically (e.g., "EndEffector")
    assembly_name = Path(input_folder).name

    # Define dynamic output paths
    out_obb = f"Outputs_{assembly_name}/OBB"
    out_aabb = f"Outputs_{assembly_name}/AABB"
    
    # =======================================================
    #                 RUN IM Generation
    # =======================================================
    print(f"\n[STARTING] AABB Pipeline for: {assembly_name}...")
    assembly_manifest_AABB = imf.load_assembly_from_folder(input_folder, bounding_box_type="AABB")
    
    start_time = time.time()
    final_matrices_AABB = imf.calculate_IM_matrices(assembly_manifest_AABB)
    
    print(f"--- AABB Math Complete! Time Taken: {(time.time() - start_time):.2f} seconds ---")
    imf.export_matrices_to_excel(final_matrices_AABB, assembly_manifest_AABB, output_folder=out_aabb)
    imf.export_directions_to_excel(assembly_manifest_AABB, output_folder=out_aabb)

    # =======================================================
    #                 RUN AND/OR Pipeline
    # =======================================================
    print(f'[STARTING] AND/OR Graph Generation')
    G = build_geometric_and_or_graph(final_matrices_AABB, assembly_manifest_AABB)
    G = build_fabrica_precedence_graph(final_matrices_AABB, assembly_manifest_AABB)
    #print_geometric_assembly_sequences(G, len(assembly_manifest_AABB))

    precedence_dir = f'Precedence_Graphs_{assembly_name}'
    save_graph(G, precedence_dir)
    

    