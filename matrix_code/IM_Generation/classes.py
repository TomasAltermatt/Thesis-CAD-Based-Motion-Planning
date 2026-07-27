import numpy as np
import networkx as nx
import pyvista as pv
import trimesh


class PseudoFace:
    def __init__(self, part: trimesh.Trimesh, face_indices: list, extraction_axis: str):
        axis_idx = {"x": 0, "y": 1, "z": 2}
        self.part = part
        self.face_indices = np.array(list(face_indices), dtype=int)
        self.extraction_axis = axis_idx[extraction_axis]
        self.facets = []
        self.focus_facets = []

        self.triangles_3d = part.triangles[self.face_indices]

        all_axes = [0, 1, 2]
        all_axes.remove(axis_idx[extraction_axis])
        self.u_axis, self.v_axis = all_axes[0], all_axes[1]
        
        # Calculate 2D coordinates by projecting 3D triangles on the extraction plane
        self.triangles_2d = self.triangles_3d[:, :, [self.u_axis, self.v_axis]]

        # Calculate the bounding box of the pseudo-face in 2D
        self.u_min = self.triangles_2d[:, :, 0].min()
        self.u_max = self.triangles_2d[:, :, 0].max()
        self.v_min = self.triangles_2d[:, :, 1].min()
        self.v_max = self.triangles_2d[:, :, 1].max()
    
    def get_focus_facets(self, SR):
        # Get the facets that intersect with center points of SR limits
        SR_min_u, SR_max_u = SR['overlap_u']
        SR_min_v, SR_max_v = SR['overlap_v']

        # Center points of the overlap limits
        center_u = (SR_min_u + SR_max_u) / 2
        center_v = (SR_min_v + SR_max_v) / 2

        # Get bounding box of each facet in 2D
        mins = self.triangles_2d.min(axis=1)  # (num_facets, 2)
        maxs = self.triangles_2d.max(axis=1)  # (num_facets, 2)

        # Get probe points
        probe_points = np.array([
            [SR_min_u, center_v],  # Left Mid
            [SR_max_u, center_v],  # Right Mid
            [center_u, SR_min_v],  # Bottom Mid
            [center_u, SR_max_v],  # Top Mid
            [center_u, center_v]   # Center
        ])

        u_hits = (mins[:, None, 0] <= probe_points[:, 0]) & (probe_points[:, 0] <= maxs[:, None, 0])
        v_hits = (mins[:, None, 1] <= probe_points[:, 1]) & (probe_points[:, 1] <= maxs[:, None, 1])

        final_mask = np.any(u_hits & v_hits, axis=1)

        self.focus_facets = np.where(final_mask)[0]
        # print(f"Focus facets indices: {self.focus_facets}")
    

    def visualize_focus_facets(self, SR, plotter, index, show_SR_box=True, show = False):
        """
        Visualizes the full part, the PseudoFace, the Focus Facets,
        the 3D Shadow Region bounding volume, and the 5 specific probe points.
        """
        # 1. Map the string extraction axis to world coordinates dynamically
        axis_idx = {"x": 0, "y": 1, "z": 2}
        w_idx = self.extraction_axis
        all_axes = [0, 1, 2]
        all_axes.remove(w_idx)
        u_idx = all_axes[0]
        v_idx = all_axes[1]

        # 2. Extract the 2D limits from the SR dictionary
        min_u, max_u = SR['overlap_u']
        min_v, max_v = SR['overlap_v']
        center_u = (min_u + max_u) / 2
        center_v = (min_v + max_v) / 2

        # Find the physical depth range of this face to extrude the box cleanly
        min_w = self.triangles_3d[:, :, w_idx].min()
        max_w = self.triangles_3d[:, :, w_idx].max()
        center_w = (min_w + max_w) / 2

        # 3. Build the 6-element PyVista bounding box array [xmin, xmax, ymin, ymax, zmin, zmax]
        pv_bounds = [0.0] * 6
        pv_bounds[2 * u_idx] = min_u
        pv_bounds[2 * u_idx + 1] = max_u
        pv_bounds[2 * v_idx] = min_v
        pv_bounds[2 * v_idx + 1] = max_v
        # Add slight padding to the extrusion depth so the box visibly slices through
        pv_bounds[2 * w_idx] = min_w - 2.0
        pv_bounds[2 * w_idx + 1] = max_w + 2.0

        # Create the 3D box representation of the Shadow Region
        sr_box = pv.Box(bounds=pv_bounds)

        # 4. Generate the 3D positions of the 5 probe points for the checker
        probes_2d = [
            [min_u, center_v],  # Left Mid
            [max_u, center_v],  # Right Mid
            [center_u, min_v],  # Bottom Mid
            [center_u, max_v],  # Top Mid
            [center_u, center_v]   # Center
        ]

        # 5. Initialize the PyVista visualizer
        mesh = pv.wrap(self.part)
        # plotter = pv.Plotter()
        
        # Draw the main part as a faint ghost
        plotter.add_mesh(mesh, color="lightgray", opacity=0.12)
        
        # Draw the entire PseudoFace in blue
        pf_mesh = mesh.extract_cells(self.face_indices)
        plotter.add_mesh(pf_mesh, color="royalblue", opacity=0.4, show_edges=True, edge_color="blue", label="PseudoFace")

        # Draw the extruded Shadow Region box as an orange wireframe cage
        if show_SR_box:
            plotter.add_mesh(sr_box, style="wireframe", color="darkorange", line_width=2, label="Shadow Region (SR)")

            # Map the 2D probe points into 3D coordinates and plot them as spheres
            for i, (p_u, p_v) in enumerate(probes_2d):
                pt_3d = [0.0, 0.0, 0.0]
                pt_3d[u_idx] = p_u
                pt_3d[v_idx] = p_v
                pt_3d[w_idx] = center_w
                
                # Label the center point uniquely, color the rest standard red
                p_color = "crimson" if i < 4 else "red"
                p_label = "Probe Points" if i == 0 else None
                plotter.add_mesh(pv.Sphere(radius=1.2, center=pt_3d), color=p_color, label=p_label)

        # Draw the selected focus facets in bright gold
        global_focus_indices = self.face_indices[self.focus_facets]
        if len(global_focus_indices) > 0:
            focus_mesh = mesh.extract_cells(global_focus_indices)
            col = 'gold' if index == 0 else 'crimson'
            plotter.add_mesh(focus_mesh, color=col, show_edges=True, edge_color="black", line_width=2, label="Focus Facets")
        
        if show:
            plotter.show()

