import bpy # pylint: disable=import-error


class BaseObj(object):

    bpy_obj: bpy.types.Object

    def __init__(self, bpy_obj: bpy.types.Object) -> None:
        self.bpy_obj = bpy_obj
        self.show() # default show

    @property
    def bpy_data(self) -> bpy.types.Mesh:
        return self.bpy_obj.data

    def show(self) -> None:
        self.bpy_obj.hide_render = False

    def hide(self) -> None:
        self.bpy_obj.hide_render = True

    def set_show(self, mode: bool = True) -> None:
        self.bpy_obj.hide_render = not mode

    def remove(self) -> None:
        bpy_obj = self.bpy_obj
        for collection in bpy_obj.users_collection:
            collection.objects.unlink(bpy_obj)
        if bpy_obj.users == 0:
            bpy.data.objects.remove(bpy_obj)
        else:
            print(f'object {bpy_obj.name} has {bpy_obj.users} users')
            bpy_obj.hide_render = True
