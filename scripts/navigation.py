import os
import pickle
import random
import numpy as np
import datasets
from PIL import Image
import math
from scripts.prompt_builder import build_action_prompt, build_viz_prompt

from scripts.action_utils import (
    DATASET_RANGES,
    DEFAULT_RANGES,
    calculate_action_delta,
    action_to_text
)


class NavigationConfig(datasets.BuilderConfig):
    def __init__(self, tasks, modes, data_dir, **kwargs):
        super(NavigationConfig, self).__init__(**kwargs)
        self.tasks = tasks
        self.modes = modes
        self.data_dir = data_dir


class NavigationDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = NavigationConfig
    BUILDER_CONFIGS = [
        NavigationConfig(
            name="processed_navigation",
            version="1.0.0",
            description="Refactored navigation dataset.",
            tasks=["navigation_simulation"],
            modes=["single_step_visualization", "action_reasoning", "task_level_evaluation"],
            data_dir="go_stanford",
        )
    ]
    DEFAULT_CONFIG_NAME = "processed_navigation"

    def _info(self):
        features = datasets.Features({
            'idx': datasets.Value('int32'),
            "input_text": datasets.Value("string"),
            "input_imgs": datasets.Sequence(datasets.Image()),
            'gt_next_action': datasets.Value("string"),
            "label_text": datasets.Value("string"),
            "label_imgs": datasets.Sequence(datasets.Image()),
            "label_img_paths": datasets.Sequence(datasets.Value("string")),
            "input_img_paths": datasets.Sequence(datasets.Value("string")),
            'task': datasets.Value('string'),
            'train_task': datasets.Value("string"),
            'coords': datasets.Sequence(datasets.Sequence(datasets.Value("float32"))), # sequence of [x,y,yaw]
            "action_vector": datasets.Sequence(datasets.Value("float32")),
        })
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        data_root = self.config.data_dir
        all_traj_names = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
        random.seed(42)
        random.shuffle(all_traj_names)

        total_used_ratio = 0.003  # Use 100% of the data, change to 0.5 to use 50%
        train_ratio, dev_ratio, test_ratio = 0.4, 0.3, 0.02
        # total_used_ratio = 0.002  # Use 100% of the data, change to 0.5 to use 50%
        # train_ratio, dev_ratio, test_ratio = 0.5, 0.3, 0.2
        total_count = int(len(all_traj_names) * total_used_ratio)
        split_point1 = int(total_count * train_ratio)
        split_point2 = split_point1 + int(total_count * dev_ratio)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={
                    "traj_dirs": [os.path.join(data_root, name) for name in all_traj_names[:split_point1]],
                    "split": datasets.Split.TRAIN 
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, 
                gen_kwargs={
                    "traj_dirs": [os.path.join(data_root, name) for name in all_traj_names[split_point1:split_point2]],
                    "split": datasets.Split.VALIDATION
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, 
                gen_kwargs={
                    "traj_dirs": [os.path.join(data_root, name) for name in all_traj_names[split_point2:total_count]],
                    "split": datasets.Split.TEST
                }
            ),
        ]

    def _load_and_validate_trajectory(self, traj_dir):
        """Loads and validates data for a single trajectory."""
        pkl_path = os.path.join(traj_dir, "traj_data.pkl")
        if not os.path.exists(pkl_path): return None

        try:
            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)
            
            positions = np.array(pkl_data['position'], dtype=float)
            yaws = np.array(pkl_data['yaw'], dtype=float).reshape(-1, 1)

            if not (positions.ndim == 2 and positions.shape[1] == 2 and yaws.ndim == 2 and yaws.shape[1] == 1 and positions.shape[0] == yaws.shape[0]):
                return None
            
            traj_len = positions.shape[0]

            image_files = sorted([f for f in os.listdir(traj_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))], key=lambda f: int(os.path.splitext(f)[0]))
            image_paths = [os.path.join(traj_dir, f) for f in image_files]

            if len(image_paths) != traj_len:
                return None

            states_xy_yaw = np.concatenate([positions, yaws], axis=1).tolist()

            return {
                "pkl_data": pkl_data,
                "image_paths": image_paths,
                "traj_len": traj_len,
                "states_xy_yaw": states_xy_yaw
            }
        except (IOError, pickle.UnpicklingError, KeyError, ValueError):
            return None

    def _prepare_actions(self, pkl_data, states_xy_yaw):
        delta_action_key = "delta"
        
        delta_actions = pkl_data.get(delta_action_key)
        if delta_actions is not None:
            delta_actions = np.array(delta_actions)
            if delta_actions.ndim == 1:
                delta_actions = delta_actions.reshape(-1, 3)
            # Removed _MAX_STEPS_PER_TRAJ logic
            return ["input_img"] + delta_actions.tolist()

        numeric_actions = [calculate_action_delta(states_xy_yaw[i], states_xy_yaw[i+1]) for i in range(len(states_xy_yaw) - 1)]
        numeric_actions.append([0.0, 0.0, 0.0])
        return ["input_img"] + numeric_actions
    

    def _get_dataset_ranges(self, image_path):
        """ Determine the dataset ranges based on the image path."""
        for name, ranges in DATASET_RANGES.items():
            if f"/{name}/" in image_path.replace("\\", "/"):
                return ranges
        return DEFAULT_RANGES
    

####Key 1
    def _prepare_visualization_sample(self, k, len_seq, all_images, all_image_paths, actions, states_xy_yaw, ranges):
        start_img = all_images[0]
        goal_img = all_images[len_seq - 1]
        current_img = all_images[k]
        next_img = all_images[k + 1] if k + 1 < len_seq else all_images[k]

        start_path = all_image_paths[0]
        goal_path = all_image_paths[len_seq - 1]
        current_path = all_image_paths[k]
        next_path = all_image_paths[k + 1] if k + 1 < len_seq else all_image_paths[k]

        current_action = action_to_text(actions[k+1]) if k < len_seq - 1 else "Stop"

        start_pose = states_xy_yaw[0]  # [x, y, yaw]
        start_pose_str = f"Starting Point Coordinate: x={start_pose[0]:.3f}, y={start_pose[1]:.3f}, yaw={start_pose[2]:.3f}\n"

        # history_action_seq_text = "\n".join([action_to_text(act) for act in actions[0:k-1]])

        # if k == len_seq - 1:
        #     next_action_text = "Stop"
        # else:
        #     next_action_text = action_to_text(actions[k + 1])
        
        input_text = build_viz_prompt(current_action, start_pose_str)

        return {
            "input_text": input_text,
            "input_imgs": [start_img, goal_img, current_img],
            "input_img_paths": [start_path, goal_path, current_path],
            "gt_next_action": f"Next Action: {current_action}",
            "label_text": "<image>",
            "label_imgs": [next_img],
            "label_img_paths": [next_path],
            "train_task": "single_step_visualization",
            "coords": states_xy_yaw[:k+1],
            "action_vector": [],
        }

# ###Key 2
    def _prepare_reasoning_sample(self, k, len_seq, all_images, all_image_paths, actions, states_xy_yaw, ranges):
        start_img = all_images[0]
        goal_img = all_images[len_seq - 1]
        current_img = all_images[k]
        next_img = all_images[k + 1] if k + 1 < len_seq else all_images[k]

        start_path = all_image_paths[0]
        goal_path = all_image_paths[len_seq - 1]
        current_path = all_image_paths[k]
        next_path = all_image_paths[k + 1] if k + 1 < len_seq else all_image_paths[k]

        start_pose = states_xy_yaw[0]  # [x, y, yaw]
        start_pose_str = f"Starting Point Coordinate: x={start_pose[0]:.3f}, y={start_pose[1]:.3f}, yaw={start_pose[2]:.3f}\n"
        
        action_history_text = ""
        for i in range(k):
            action_history_text += f"{action_to_text(actions[i+1])}"
        
        dxy_range = ranges['dxy']
        dyaw_range = ranges['dyaw']
        input_text = build_action_prompt(start_pose_str, dxy_range, dyaw_range)
        
        return {
            "input_text": input_text,
            "input_imgs": [start_img, goal_img, current_img],
            "input_img_paths": [start_path, goal_path, current_path],
            "gt_next_action": "",
            "label_text": action_to_text(actions[k+1]) if k < len_seq - 1 else "Stop",
            "label_imgs": [],
            "label_img_paths": [],
            "train_task": "action_reasoning",
            "coords": states_xy_yaw[:k+1],
            "action_vector": actions[k+1] if k < len_seq - 1 else [0.0, 0.0, 0.0],
        }
    
    def _prepare_task_level_sample(self, len_seq, all_images, all_image_paths, actions, states_xy_yaw, ranges):
        start_img = all_images[0]
        goal_img = all_images[len_seq - 1]
        start_path = all_image_paths[0]
        goal_path = all_image_paths[len_seq - 1]
        start_pose = states_xy_yaw[0]

        start_pose_str = f"Starting Point Coordinate: x={start_pose[0]:.3f}, y={start_pose[1]:.3f}, yaw={start_pose[2]:.3f}\n"

        input_text = (
            "Task: Full Navigation Task\n"
            "Description: Navigate from the start observation to the goal observation. "
            "Start Observation: <image>\nGoal Observation: <image>\n"
            f"{start_pose_str}"
        )

        # The 'label' for this task is the final goal image, to be compared against the final predicted image.
        # We also pass the ground truth trajectory for more detailed analysis if needed.
        return {
            "input_text": input_text,
            "input_imgs": [start_img, goal_img],
            "input_img_paths": [start_path, goal_path],
            "label_text": "",
            "label_imgs": [goal_img],
            "label_img_paths": [goal_path],
            "train_task": "task_level_evaluation",
            "coords": states_xy_yaw,
            "action_vector": [],
            "ranges": ranges,
        }

    def _generate_examples(self, traj_dirs, split):
        print(f"Current config modes: {self.config.modes}")
        global_idx = 0
        for traj_dir in traj_dirs:
            # 1. Load and validate data for one trajectory
            traj_data = self._load_and_validate_trajectory(traj_dir)
            if not traj_data:
                continue
            trajectory_ranges = self._get_dataset_ranges(traj_data["image_paths"][0])

            # 2. Prepare the definitive list of actions for this trajectory
            actions = self._prepare_actions(
                traj_data["pkl_data"],
                traj_data["states_xy_yaw"],
            )

            # 3. Load all images for the trajectory
            try:
                all_images = [Image.open(p).convert("RGB").resize((256, 256)) for p in traj_data["image_paths"]]
            except IOError:
                continue
            
            if "task_level_evaluation" in self.config.modes and split == datasets.Split.TEST:
                print(1)
                sample = self._prepare_task_level_sample(len(all_images), all_images,
                                                         traj_data["image_paths"], actions,
                                                         traj_data["states_xy_yaw"], trajectory_ranges)
                sample['task'] = self.config.tasks[0]
                sample['idx'] = global_idx
                yield global_idx, sample
                global_idx += 1

            # Use the full length of all_images directly
            # 4. Generate samples
            for k in range(len(all_images)):
                # Generate a sample for the reasoning task
                if "action_reasoning" in self.config.modes and split != datasets.Split.TEST:
                    sample = self._prepare_reasoning_sample(k, len(all_images), all_images,
                                                            traj_data["image_paths"], actions,
                                                            traj_data["states_xy_yaw"], trajectory_ranges)

                    sample['task'] = self.config.tasks[0]
                    sample['idx'] = global_idx
                    yield global_idx, sample
                    global_idx += 1
                    

                # Generate a sample for the visualization task
                if "single_step_visualization" in self.config.modes and split != datasets.Split.TEST:
                    sample = self._prepare_visualization_sample(k, len(all_images), all_images,
                                                                traj_data["image_paths"], actions,
                                                                traj_data["states_xy_yaw"], trajectory_ranges)
                    sample['task'] = self.config.tasks[0]
                    sample['idx'] = global_idx
                    yield global_idx, sample
                    global_idx += 1