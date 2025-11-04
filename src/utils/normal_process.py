import numpy as np
import json
import matplotlib.pyplot as plt 
 
 
 
 
def get_transform_matrix(json_file_path):
        """
        Read a JSON file and return the transform_matrix for the first frame.

        :param json_file_path: Path to the JSON file
        :return: 4x4 transform_matrix (NumPy array) or None if not found
        """
        with open(json_file_path, "r") as f:
            data = json.load(f)

        for frame_list in data["frames"]:
            return np.array(frame_list.get('transform_matrix'))
            
def blender_world_normal_2_opengl_camera(normals_world: np.ndarray, c2w: np.ndarray, visualization = False) -> np.ndarray:    
        H, W, C = normals_world.shape
        if C == 4:
            normals_world = normals_world[..., :3]

        R_c2w = c2w[:3, :3]
        R_opencv = R_c2w.T

        transformed_normals = normals_world.reshape(-1, 3).T  
        transformed_normals = R_opencv @ transformed_normals
        transformed_normals = transformed_normals.T
        transformed_normals = transformed_normals.reshape(H, W, 3)
        if visualization:
            plt.imshow(transformed_normals)
            plt.axis('off')
            plt.savefig('N_world.png',bbox_inches='tight', pad_inches=0)
            transformed_normals = transformed_normals * 0.5 + 0.5
            plt.imshow(transformed_normals)
            plt.axis('off')
            plt.savefig('N_camera.png',bbox_inches='tight', pad_inches=0)
        return transformed_normals