from dataclasses import dataclass


@dataclass
class Instance:
    """Class for track examples from database"""
    instance_text: list
    instance_label: str

    def __init__(self, instance_text: list, instance_label: str):
        self.instance_text = instance_text
        self.instance_label = instance_label
