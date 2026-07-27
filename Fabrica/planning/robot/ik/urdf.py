import xml.etree.ElementTree as ET
import json
import numpy as np
import itertools
import warnings

from ikpy import link as lib_link
from ikpy import logs
from ikpy.urdf.URDF import _find_next_joint, _find_next_link


def get_urdf_parameters(urdf_file, base_elements=None, last_link_vector=None, base_element_type="link", symbolic=True, scale_translation=1.0, origin_translation=None, origin_orientation=None, reduced_limit=0.0):
    """
    Returns translated parameters from the given URDF file.
    Parse the URDF joints into IKPY links, throw away the URDF links.

    Parameters
    ----------
    urdf_file: str
        The path of the URDF file
    base_elements: list of strings
        List of the links beginning the chain
    last_link_vector: numpy.array
        Optional : The translation vector of the tip.
    base_element_type: str
    symbolic: bool

    Returns
    -------
    list[ikpy.link.URDFLink]
    """
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    base_elements = list(base_elements)
    if base_elements is None:
        base_elements = ["base_link"]
    elif base_elements is []:
        raise ValueError("base_elements can't be the empty list []")

    joints = []
    links = []
    has_next = True
    current_joint = None
    current_link = None
    joint_link_map = {}

    # Initialize the tree traversal
    if base_element_type == "link":
        # The first element is a link, so its (virtual) parent should be a joint
        node_type = "joint"
    elif base_element_type == "joint":
        # The same as before, but swap link and joint
        node_type = "link"
    else:
        raise ValueError("Unknown type: {}".format(base_element_type))

    # Parcours récursif de la structure de la chain
    while has_next:
        if len(base_elements) != 0:
            next_element = base_elements.pop(0)
        else:
            next_element = None

        if node_type == "link":
            # Current element is a link, find child joint
            (has_next, current_joint) = _find_next_joint(root, current_link, next_element)
            node_type = "joint"
            if has_next:
                joints.append(current_joint)
                logs.logger.debug("Next element: joint {}".format(current_joint.attrib["name"]))

        elif node_type == "joint":
            # Current element is a joint, find child link
            (has_next, current_link) = _find_next_link(root, current_joint, next_element)
            node_type = "link"
            if has_next:
                links.append(current_link)
                logs.logger.debug("Next element: link {}".format(current_link.attrib["name"]))
                if current_joint is None:
                    assert None not in joint_link_map, "Multiple base links"
                    joint_link_map[None] = current_link.attrib["name"]
                else:
                    joint_link_map[current_joint.attrib["name"]] = current_link.attrib["name"]

    parameters = []

    if None in joint_link_map:
        parameters.append(lib_link.URDFLink(
            name=joint_link_map[None],
            bounds=tuple([-np.inf, np.inf]),
            origin_translation=origin_translation if origin_translation is not None else np.array([0, 0, 0]),
            origin_orientation=origin_orientation if origin_orientation is not None else np.array([0, 0, 0]),
            rotation=None,
            translation=None,
            use_symbolic_matrix=symbolic,
            joint_type='fixed'
        ))

    # Save the joints in the good format
    for joint in joints:
        origin_translation = [0, 0, 0]
        origin_orientation = [0, 0, 0]
        rotation = None
        translation = None
        bounds = [-np.inf, np.inf]

        origin = joint.find("origin")
        if origin is not None:
            if "xyz" in origin.attrib.keys():
                origin_translation = [float(x) for x in origin.attrib["xyz"].split()]
            if "rpy" in origin.attrib.keys():
                origin_orientation = [float(x) for x in origin.attrib["rpy"].split()]

        joint_type = joint.attrib["type"]
        if joint_type not in ["revolute", "prismatic", "fixed"]:
            raise ValueError("Unknown joint type: {}".format(joint_type))

        axis = joint.find("axis")
        if axis is not None:
            if joint_type == "revolute":
                rotation = [float(x) for x in axis.attrib["xyz"].split()]
                translation = None
            elif joint_type == "prismatic":
                rotation = None
                translation = [float(x) for x in axis.attrib["xyz"].split()]
            elif joint_type == "fixed":
                pass
                # warnings.warn("Joint {} is of type: fixed, but has an 'axis' attribute defined. This is not in the URDF spec and thus this axis is ignored".format(joint.attrib["name"]))
            else:
                raise ValueError("Unknown joint type with an axis: {}, {}".format(joint_type, axis))

        limit = joint.find("safety_controller")
        if limit is not None:
            if "soft_lower_limit" in limit.attrib:
                bounds[0] = float(limit.attrib["soft_lower_limit"])
            if "soft_upper_limit" in limit.attrib:
                bounds[1] = float(limit.attrib["soft_upper_limit"])
        else:
            limit = joint.find("limit")
            if limit is not None:
                if "lower" in limit.attrib:
                    bounds[0] = float(limit.attrib["lower"])
                if "upper" in limit.attrib:
                    bounds[1] = float(limit.attrib["upper"])
        
        if reduced_limit > 0.0:
            reduced_amount = (bounds[1] - bounds[0]) * reduced_limit
            bounds[0] += reduced_amount
            bounds[1] -= reduced_amount

        parameters.append(lib_link.URDFLink(
            name=joint_link_map[joint.attrib["name"]],
            bounds=tuple(bounds),
            origin_translation=np.array(origin_translation) * scale_translation,
            origin_orientation=origin_orientation,
            rotation=rotation,
            translation=translation if translation is None else np.array(translation) * scale_translation,
            use_symbolic_matrix=symbolic,
            joint_type=joint_type
        ))

    # Add last_link_vector to parameters
    if last_link_vector is not None:
        # The last link doesn't provide a rotation
        parameters.append(lib_link.URDFLink(
            origin_translation=np.array(last_link_vector) * scale_translation,
            origin_orientation=[0, 0, 0],
            rotation=None,
            translation=None,
            name="last_joint",
            use_symbolic_matrix=symbolic,
            joint_type="fixed"
        ))

    return parameters
