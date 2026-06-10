import numpy as np

class BBox2D:
    def __init__(self, x_min, y_min, x_max, y_max, semantic_id, obj_name):
        self.x_min, self.y_min, self.x_max, self.y_max = x_min, y_min, x_max, y_max
        self.semantic_id = semantic_id
        self.obj_name = obj_name

    def to_array(self):
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max])

