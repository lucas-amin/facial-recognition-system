import json
import os
import tempfile

from json_file_wrapper import JSONFileWrapper


def tmp_path():
    return tempfile.TemporaryDirectory()


def test_append_and_save_new_file(tmp_path):
    file_path = tmp_path / "test_data.json"
    wrapper = JSONFileWrapper(file_path)
    assert len(wrapper.json_data) == 0  # File shouldn't exist yet

    wrapper.append_and_save("item1")
    wrapper.append_and_save("item2")
    wrapper.append_and_save({"key": "value"})

    assert os.path.exists(file_path)
    with open(file_path, 'rb') as f:
        data = json.load(f)
        assert data == ["item1", "item2", {"key": "value"}]


def test_load_and_append(tmp_path):
    file_path = tmp_path / "test_data.json"

    # Write initial data using json.dump with 'w' mode (text)
    with open(file_path, 'w') as f:
        json.dump(["existing_item"], f)

    wrapper = JSONFileWrapper(file_path)
    assert wrapper.json_data == ["existing_item"]  # Use json_data, not pickle_data

    wrapper.append_and_save("new_item")

    # Read data using json.load with 'r' mode (text)
    with open(file_path, 'r') as f:
        data = json.load(f)
        assert data == ["existing_item", "new_item"]


def test_empty_file(tmp_path):
    file_path = tmp_path / "test_data.json"

    # Create an empty file using 'w' mode
    with open(file_path, 'w'):
        pass

    wrapper = JSONFileWrapper(file_path)
    assert wrapper.json_data == []
    wrapper.append_and_save("item1")
    assert wrapper.json_data == ["item1"]
