from synth_ndf_dataset.utils import _passenger_key


class Joint:
    def __init__(self, content: dict):
        self.name = content.get("name", "")
        self.index = content["index"]
        self.parent_index = content.get("parent_index")
        self.world = content.get("world")

class Skeleton:
    def __init__(self, content: dict):
        self.joints = [Joint(j) for j in content["joints"]]
        self.idx_to_joint = {j.index: j for j in self.joints}
        self.prim_path = content.get("prim_path", "")

    def joint(self, key):
        if isinstance(key, str):
            for j in self.joints:
                if j.name == key:
                    return j
            return None
        elif isinstance(key, int):
            return self.idx_to_joint.get(key)
        else:
            return None

    def get_parent(self, key):
        joint = self.joint(key)
        if joint is None:
            return None
        parent_index = joint.parent_index
        if parent_index is None:
            return None
        return self.joint(parent_index)

    def connections(self, with_root=False):
        connections = []
        for joint in self.joints:
            parent = self.get_parent(joint.index)
            if parent is not None:
                if not with_root and parent.parent_index < 0:
                    continue
                connections.append((parent.index, joint.index))
        return connections

    def passenger_key(self):
        return _passenger_key(self.prim_path)