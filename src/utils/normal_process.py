import numpy as np
import json
import matplotlib.pyplot as plt 
 
 
 
 
def get_transform_matrix(json_file_path):
        """
        根据 file_path 从 JSON 文件中查找对应的 transform_matrix。

        :param json_file_path: JSON 文件路径
        :param target_file_path: 目标 file_path，用于匹配
        :return: 4×4 transform_matrix (NumPy 数组) 或 None（如果未找到）
        """
        # 读取 JSON 文件
        # name_prefix = target_file_prefix + '_hdri' #光照不同不会影响相机参数，所以只取0
        with open(json_file_path, "r") as f:
            data = json.load(f)

        # 遍历 JSON，查找目标 file_path 的 transform_matrix
        for frame_list in data["frames"]:
            return np.array(frame_list.get('transform_matrix')) # 遍历所有 HDRI 组
            
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