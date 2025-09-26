# scripts/trainer/task_eval_utils/prompt_builder.py
import json
from typing import Dict, Tuple


def build_action_prompt(start_pose_str: str, dxy_range: Tuple[float, float], dyaw_range: Tuple[float, float]) -> str:
    return (
        "Task: Navigation Action Prediction\n"
        "Based on the current first-person observation, starting point observation and coordinate, goal point observation, predict the next action to take. The definition of actions is as follows.\n"
        "Action Definitions: \n"
        "The action can be the language command 'Stop', indicating the end of the trajectory. Alternatively, the action can be shifts composed of three components:\n"
        "- dx: displacement along the agent's facing direction),\n"
        "- dy: displacement perpendicular to the facing direction),\n"
        "- dyaw: change in heading angle (i.e., how much the agent rotates).\n"
        "All components are discretized into bin tokens: for example,\n"
        "- `dx pos bin 02`: dx = +0.02 meters,\n"
        "- `dy neg bin 23`: dy = -0.23 meters,\n"
        "- `dyaw pos bin 26`: counterclockwise rotation of +0.26 radians.\n"
        "If the agent reaches the goal or believes it has reached, it should predict 'Stop'.\n"
        f"Action Format: \n"
        f"-Range of dx, dy: [{dxy_range[0]:.2f}, {dxy_range[1]:.2f}], -Range of dyaw: [{dyaw_range[0]:.2f}, {dyaw_range[1]:.2f}]. -Output format: Move by dx: <dx>, dy: <dy>, dyaw: <dyaw>\n"
        "Inputs:\n"
        "- Start Observation: <image> \n"
        "- Goal Observation: <image> \n"
        "- Current Observation: <image> \n"
        f"{start_pose_str} \n" 
        "Goal: \n"
        "Predict the next action to approach the goal observation"
    )

def build_viz_prompt(decoded_action: str, start_pose_str: str) -> str:
    return (
        "Task: Navigation Single Step Visualization\n"
        "Description: Given the current first-person observation, predict the next first-person view observation after the agent executes a specified navigation action.\n To assist your prediction, you may refer to the start observation and pose (position: x, y and heading: yaw), as well as the goal and current observation.\n"
        "Inputs:\n"
        f"Next Action: {decoded_action}.\n"
        f"{start_pose_str} \n"
        "- Start Observation: <image> \n"
        "- Goal Observation: <image> \n"
        "- Current Observation: <image> \n"
        "Action Format:\n"
        "The action can be the language command 'Stop', indicating the end of the trajectory. Alternatively, the action can be shifts composed of three components:\n"
        "- dx: displacement along the agent's facing direction),\n"
        "- dy: displacement perpendicular to the facing direction),\n"
        "- dyaw: change in heading angle (i.e., how much the agent rotates).\n"
        "All components are discretized into bin tokens: for example,\n"
        "- `dx pos bin 02`: dx = +0.02 meters,\n"
        "- `dy neg bin 23`: dy = -0.23 meters,\n"
        "- `dyaw pos bin 26`: counterclockwise rotation of +0.26 radians.\n"
        "Spatial Interpretation:\n"
        "- The magnitude of [dx, dy] reflects how far the agent moves in this step — larger values indicate greater positional shift, leading to larger visual changes \n"
        "- dyaw controls the agent's rotation (change in heading). A positive dyaw indicates a left turn (counter-clockwise), while a negative dyaw indicates a right turn (clockwise). \n"
        "Goal: \n"
        "Predict the most likely next first-person observation, considering how the movement and rotation implied by `dx`, `dy`, and `dyaw` would affect what the agent sees next."
    )