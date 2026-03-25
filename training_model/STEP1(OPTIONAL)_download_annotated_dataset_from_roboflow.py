"""Download the dataset from Roboflow"""
from roboflow import Roboflow

ROBOFLOW_API_KEY = "your_api_key_here"
ROBOFLOW_WORKSPACE = "your_workspace_here"
ROBOFLOW_PROJECT = "your_project_here"
ROBOFLOW_VERSION = 10

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
version = project.version(ROBOFLOW_VERSION)
dataset = version.download("yolov8")
