import trimesh


def create_manager_from_manager(manager, object_names):
    target_manager = trimesh.collision.CollisionManager()
    for object_name in object_names:
        obj = manager._objs[object_name]
        target_manager._objs[object_name] = obj
        target_manager._names[id(obj['geom'])] = object_name
        target_manager._manager.registerObject(obj['obj'])
        target_manager._manager.update()
    return target_manager
