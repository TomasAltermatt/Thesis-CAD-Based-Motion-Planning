import numpy as np
from scipy.spatial.transform import Rotation as R
from ikpy.chain import Chain as IKPyChain
from .inverse_kinematics import inverse_kinematic_optimization
from .urdf import get_urdf_parameters
from ikpy import link as link_lib


class Chain(IKPyChain):

    no_collision_links = [] # List of tuples of links that can be ignored for collision checking

    def active_to_full(self, active_joints, initial_position=None):
        if initial_position is None:
            initial_position = [0] * len(self.links)
        return super().active_to_full(active_joints, initial_position)

    def forward_kinematics_active(self, angles, **kwargs):
        """Computes the forward kinematics on the active joints of the chain

        Parameters
        ----------
        angles: numpy.array
            The angles of the active joints of the chain

        Returns
        -------
        numpy.array:
            The transformation matrix of the end effector
        """
        return self.forward_kinematics(self.active_to_full(angles), **kwargs)

    def check_success(self, actual_q, target, orientation_mode):

        ef_actual_matrix = self.forward_kinematics(actual_q)
        if orientation_mode == 'X':
            actual_ori = ef_actual_matrix[:3, 0]
            target_ori = target[:3, 0]
            diff_ori = 1 - np.dot(actual_ori, target_ori)
        elif orientation_mode == 'Y':
            actual_ori = ef_actual_matrix[:3, 1]
            target_ori = target[:3, 1]
            diff_ori = 1 - np.dot(actual_ori, target_ori)
        elif orientation_mode == 'Z':
            actual_ori = ef_actual_matrix[:3, 2]
            target_ori = target[:3, 2]
            diff_ori = 1 - np.dot(actual_ori, target_ori)
        elif orientation_mode == 'all':
            actual_ori = ef_actual_matrix[:3, :3]
            target_ori = target[:3, :3]
            relative_ori = actual_ori.T @ target_ori
            diff_ori = 1 - np.trace(relative_ori) / 3
        else:
            raise NotImplementedError
        
        diff_pos = np.linalg.norm(ef_actual_matrix[:3, 3] - target[:3, 3])
        return diff_pos < 1e-4 and diff_ori < 1e-6
    
    def inverse_kinematics_above_ground(self, target_position=None, target_orientation=None, orientation_mode=None, initial_position=None, ground_z=0, n_restart=3, **kwargs):
        solution, success = None, False
        n_trial = 0
        while not success and n_trial < n_restart:
            solution, success = super().inverse_kinematics(target_position, target_orientation, orientation_mode, initial_position=initial_position, **kwargs)
            fks = self.forward_kinematics(solution, full_kinematics=True)
            fks = [fks[link_idx] for link_idx in range(len(self.links)) if self.active_links_mask[link_idx]]
            success = True
            for fk in fks:
                if fk[2, 3] < ground_z:
                    success = False
                    break
            n_trial += 1
            initial_position = self.active_to_full(self.sample_joint_angles_active(), initial_position)
        return solution, success

    def inverse_kinematics_frame(self, target, initial_position=None, n_restart=3, **kwargs):
        """Computes the inverse kinematic on the specified target

        Parameters
        ----------
        target: numpy.array
            The frame target of the inverse kinematic, in meters. It must be 4x4 transformation matrix
        initial_position: numpy.array
            Optional : the initial position of each joint of the chain. Defaults to 0 for each joint
        kwargs: See ikpy.inverse_kinematics.inverse_kinematic_optimization

        Returns
        -------
        list:
            The list of the positions of each joint according to the target. Note : Inactive joints are in the list.
        """
        # Checks on input
        target = np.array(target)
        if target.shape != (4, 4):
            raise ValueError("Your target must be a 4x4 transformation matrix")

        if initial_position is None:
            initial_position = self.active_to_full(self.rest_q)
        initial_position = np.array(initial_position)

        success = False
        n_optimized = 0
        assert n_restart >= 1 or n_restart == -1 # -1 means infinite looping
        while not success and ((n_optimized < n_restart and n_restart >= 1) or n_restart == -1):
            solution, success = inverse_kinematic_optimization(self, target, starting_nodes_angles=initial_position, **kwargs)
            success = self.check_success(solution, target, kwargs['orientation_mode'])
            n_optimized += 1
            # initial_position = self.active_to_full(self.sample_joint_angles_active(), initial_position)
            initial_position = self.add_noise_to_joint_angles(initial_position)
        return solution, success
    
    def add_noise_to_joint_angles(self, joint_angles, noise_level=0.1):
        new_joint_angles = joint_angles.copy()
        for link_idx, link in enumerate(self.links):
            if self.active_links_mask[link_idx]:
                new_joint_angles[link_idx] += np.random.uniform(-noise_level, noise_level)
                new_joint_angles[link_idx] = np.clip(new_joint_angles[link_idx], link.bounds[0], link.bounds[1])
        return new_joint_angles

    def sample_joint_angles(self):

        joint_angles = []

        for link in self.links:
            if link.joint_type != 'fixed':
                joint_angle = np.random.uniform(link.bounds[0], link.bounds[1])
                joint_angles.append(joint_angle)
            else:
                joint_angles.append(0)

        return np.array(joint_angles)

    def sample_joint_angles_active(self):

        joint_angles = []

        for link_idx, link in enumerate(self.links):
            if self.active_links_mask[link_idx]:
                joint_angle = np.random.uniform(link.bounds[0], link.bounds[1])
                joint_angles.append(joint_angle)

        return np.array(joint_angles)
    
    def get_active_link_bounds(self):
        return [link.bounds for link in self.links if link.joint_type != 'fixed']
    
    def get_active_link(self, link_idx):
        active_link_idx = 0
        for link in self.links:
            if link.joint_type != 'fixed':
                if active_link_idx == link_idx:
                    return link
                active_link_idx += 1
        return None
    
    def get_base_link_name(self):
        assert self.links[0].joint_type == 'fixed'
        return self.links[0].name
    
    def get_eef_link_name(self):
        for link in reversed(self.links):
            if link.joint_type != 'fixed':
                return link.name
        return None

    def check_neighboring_links(self, linka_name, linkb_name):

        all_link_names = [link.name for link in self.links]
        linka_idx = all_link_names.index(linka_name)
        linkb_idx = all_link_names.index(linkb_name)

        if linka_idx == linkb_idx - 1 or linka_idx == linkb_idx + 1:
            return True
        else:
            return False
        
    def check_colliding_links(self, linka_name, linkb_name):
        if (linka_name, linkb_name) in self.no_collision_links or (linkb_name, linka_name) in self.no_collision_links:
            return False
        return not self.check_neighboring_links(linka_name, linkb_name)
        
    @classmethod
    def from_urdf_file(cls, urdf_file, base_elements, last_link_vector=None, base_element_type="link", name="chain", symbolic=True, 
        origin_translation=None, origin_orientation=None, scale_translation=1.0, reduced_limit=0.0):
        """Creates a chain from an URDF file

        Parameters
        ----------
        urdf_file: str
            The path of the URDF file
        base_elements: list of strings
            List of the links beginning the chain
        last_link_vector: numpy.array
            Optional : The translation vector of the tip.
        name: str
            The name of the Chain
        base_element_type: str
        symbolic: bool
            Use symbolic computations


        Note
        ----
        IKPY works with links, whereras URDF works with joints and links. The mapping is currently misleading:

        * URDF joints = IKPY links
        * URDF links are not used by IKPY. They are thrown away when parsing
        """

        # FIXME: Rename links to joints, to be coherent with URDF?
        urdf_metadata = {
            "base_elements": base_elements,
            "urdf_file": urdf_file,
            "last_link_vector": last_link_vector
        }

        links = get_urdf_parameters(urdf_file, base_elements=base_elements, last_link_vector=last_link_vector, base_element_type=base_element_type, symbolic=symbolic, scale_translation=scale_translation,
                                    origin_translation=origin_translation, origin_orientation=origin_orientation, reduced_limit=reduced_limit)

        active_links_mask = [False if link.joint_type == 'fixed' else True for link in links]
        chain = cls(links, active_links_mask=active_links_mask, name=name, urdf_metadata=urdf_metadata)

        # Save some useful metadata
        # FIXME: We have attributes specific to objects created in this style, not great...
        chain.urdf_file = urdf_file
        chain.base_elements = base_elements

        return chain
