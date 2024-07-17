import json
import logging
import os


class JSONFileWrapper:
    def __init__(self, path):
        self.path = path
        self.json_data = []
        if os.path.exists(self.path):
            self.load()
        else:
            self.save()

    def save(self):
        logging.info(f"Saving data into file {self.path}")
        with open(self.path, 'w') as f:
            json.dump(self.json_data, f, indent=4)  # Pretty-print JSON

    def load(self):
        with open(self.path, 'r') as f:
            try:
                self.json_data = json.load(f)
            except json.JSONDecodeError:
                pass  # Keep self.json_data as an empty list
            return self.json_data

    def append_and_save(self, item):
        self.json_data.append(item)
        self.save()
