from fastapi import FastAPI, HTTPException, Body, Response
from fastapi.middleware.cors import CORSMiddleware
from nbformat import from_dict, write, read
from nbconvert.preprocessors import ExecutePreprocessor

from typing import Dict
import json
import os

app = FastAPI()
origins = [
    "https://codebox.isolutionsai.com",
    # add more origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def check_if_file_exists(file_path: str):
    if os.path.isfile(file_path):
        return True
    else:
        return False

def execute_notebook(notebook_path):
    with open(notebook_path) as f:
        nb = read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    try:
        ep.preprocess(nb)
    except Exception as e:
        raise e

    return nb

def get_output_from_cell(nb, cell_index):
    cell = nb.cells[cell_index]
    if cell.cell_type == 'code':
        return cell.outputs
    else:
        return None

@app.post("/list-files/")
async def list_files():
    try:
        files = os.listdir("/home/ubuntu/automatenb/environment/")
        return {"status": "success", "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read-notebook/{filename}", response_class=Response)
async def read_notebook(filename: str):
    notebook_path = f"/home/ubuntu/automatenb/environment/{filename}"
    try:
        with open(notebook_path) as f:
            nb = read(f, as_version=4)

        # Convert the notebook to a dictionary
        nb_dict = dict(nb)

        # Convert the dictionary to a JSON string
        nb_json = json.dumps(nb_dict)

        # Return the JSON string as a stream
        return Response(content=nb_json, media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute-notebook/")
#async def execute(notebook_path: str, cell_index: int):
async def execute(notebook_path: str = Body(...), cell_index: int = Body(...)):
    try:
        nb = execute_notebook(notebook_path)
        output = get_output_from_cell(nb, cell_index)
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/write-notebook/")
async def write_notebook(notebook_data: Dict):
    notebook_path = notebook_data.get('path')
    if check_if_file_exists(notebook_path):
        return {"status": "error", "message": "File already exists"}
    try:
        notebook_path = notebook_data.get('path')
        notebook_content = notebook_data.get('content')

        # Convert the dictionary to a notebook object
        nb = from_dict(notebook_content)

        # Write the notebook node to a file
        with open(notebook_path, 'w') as f:
            write(nb, f)

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/notebook-environment/")
async def notebook_environment(notebook_data: Dict, folder_name: str = Body(...)):
    # Create a new directory with the folder_name as its name
    os.makedirs(folder_name, exist_ok=True)

    # Get the notebook path and content from the request data
    notebook_path = notebook_data.get('path')
    notebook_content = notebook_data.get('content')

    # Append the new folder name to the notebook path
    notebook_path = os.path.join(folder_name, notebook_path)

    if check_if_file_exists(notebook_path):
        return {"status": "error", "message": "File already exists"}

    try:
        # Convert the dictionary to a notebook object
        nb = from_dict(notebook_content)

        # Write the notebook node to a file
        with open(notebook_path, 'w') as f:
            write(nb, f)

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete-file/")
async def delete_file(filename: str = Body(...)):
    filename= json.loads(filename)['filename']
    file_path = f"/home/ubuntu/automatenb/environment/{filename}"
    print(file_path)
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
            return {"status": "success", "message": f"File {filename} has been deleted."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        return {"status": "error", "message": "File does not exist"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
