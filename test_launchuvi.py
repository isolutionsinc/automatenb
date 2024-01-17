from fastapi.testclient import TestClient
from launchuvi import app, notebook_environment
import pytest

client = TestClient(app)

def test_notebook_environment():
    # Define a valid notebook data dictionary
    notebook_data = {
        "path": "test_notebook.ipynb",
        "content": {
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": []
                }
            ]
        }
    }

    # Define a valid folder name
    folder_name = "test_create_folder"

    # Make a POST request to the notebook_environment endpoint
    response = client.post("/notebook-environment/", json={"notebook_data": notebook_data, "folder_name": folder_name})

    # Assert that the response status code is 200 (OK)
    assert response.status_code == 200

    # Assert that the response JSON matches the expected result
    assert response.json() == {"status": "success"}

def test_list_files():
    response = client.post("/list-files/", json={"folder_name": "test_folder"})
    assert response.status_code == 200
    assert "files" in response.json()


def test_read_notebook():
    response = client.get("/read-notebook/test_folder/test_notebook.ipynb")
    assert response.status_code == 200
    assert "cells" in response.json()