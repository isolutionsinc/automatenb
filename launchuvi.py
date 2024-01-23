from fastapi import FastAPI, HTTPException, Body, Response, File, UploadFile, Query, Depends, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from nbformat import from_dict, write, read
from nbformat.v4 import new_code_cell, new_markdown_cell
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
from nbconvert import PythonExporter, NotebookExporter

from typing import Dict, List, Optional
import json
from dotenv import load_dotenv
import os
import shutil
import subprocess
import traceback
import logging
import sys
import requests

load_dotenv()

from supabase import create_client, Client
from starlette.responses import StreamingResponse
import aiohttp
import aiofiles
from fastapi import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST

from pydantic import BaseModel
from typing import Any

TOKEN = os.getenv("TOKEN")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def verify_token(token: str = Depends(oauth2_scheme)):
    if token != TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

class FolderName(BaseModel):
    folder_name: str

class ReplaceNotebookData(BaseModel):
    folder_name: str
    file_name: str
    content: Any

class DeleteFileInput(BaseModel):
    folder_name: str
    file_name: str

class ExecuteCellInput(BaseModel):
    folder_name: str
    file_name: str
    cell_index: int

class DownloadFileInput(BaseModel):
    url: str
    folder_name: str
    file_name: str

class ExecuteNotebook(BaseModel):
    folder_name: str
    file_name: str

class DLFileData(BaseModel):
    bucket_name: str
    folder_name: str
    file_name: str
    expiry_duration: int

class ULFileData(BaseModel):
    bucket_name: str
    folder_name: str
    file_name: str
    expiry_duration: int

class Cell(BaseModel):
    cell_type: Optional[str] = "code"
    execution_count: Optional[None] = None
    metadata: Optional[dict] = {}
    outputs: Optional[List] = []
    source: Optional[List[str]] = [""]

class UpdateCellInput(BaseModel):
    file_name: str
    folder_name: str
    cell_index: int
    cell_content: str
    cell_type: str

class DeleteCellInput(BaseModel):
    file_name: str
    folder_name: str
    cell_index: int

class Content(BaseModel):
    cells: List[Cell]
    metadata: Optional[dict] = {}
    nbformat: Optional[int] = 4
    nbformat_minor: Optional[int] = 4

class NotebookData(BaseModel):
    content: Content

class Notebook(BaseModel):
    file_name: Optional[str] = "default.ipynb"
    folder_name: Optional[str] = "default"
    notebook_data: NotebookData


url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(url, key)

print(url)
print(key)

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

def get_output_from_cell(nb, cell_index):
    cell = nb.cells[cell_index]
    if cell.cell_type == 'code':
        return cell.outputs
    else:
        return None

class FolderName(BaseModel):
    folder_name: str

def handle_input(notebook: Notebook):
    return notebook

@app.post("/list-files")
async def list_files(input: FolderName, token: str = Depends(verify_token)):
    try:
        folder_path = f"/home/ubuntu/automatenb/environment/{input.folder_name}"
        files = os.listdir(folder_path)
        return {"status": "success", "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/read-notebook", response_class=Response)
async def read_notebook(folder_name: str = Body(...), file_name: str = Body(...), token: str = Depends(verify_token)):
    notebook_path = f"/home/ubuntu/automatenb/environment/{folder_name}/{file_name}"
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

def execute_notebook(notebook_path, working_dir, cell_index):
    with open(notebook_path) as f:
        nb = read(f, as_version=4)

    # Add a code cell at the beginning of the notebook that changes the working directory
    change_dir_cell = new_code_cell(source=f'import os\nos.chdir(r"{working_dir}")')
    nb.cells.insert(0, change_dir_cell)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    try:
        ep.preprocess(nb)
    except CellExecutionError as e:
        tb_str = traceback.format_exc()
        return {"status": "error", "message": "Error executing cell", "traceback": tb_str}

    # Remove the change_dir_cell after execution
    nb.cells.pop(0)

    # Get the output from the cell at cell_index
    #output = nb.cells[cell_index + 1].outputs
    output = nb.cells[cell_index].outputs

    return {"notebook": nb, "output": output}

@app.post("/execute-notebook")
async def execute(input: ExecuteNotebook, token: str = Depends(verify_token)):
    try:
        folder_path = f"/home/ubuntu/automatenb/environment/{input.folder_name}"
        notebook_path = f"{folder_path}/{input.file_name}"
        cell_index = 0
        result = execute_notebook(notebook_path, folder_path, cell_index)
        if "status" in result and result["status"] == "error":
            return result
        return {"notebook": result["notebook"], "output": result["output"]}
    except Exception as e:
        logging.error(f"Error executing cell: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/execute-cell-as-script")
# async def execute_cell_as_script(input: ExecuteCellInput, token: str = Depends(verify_token)):
#     try:
#         folder_path = f"/home/ubuntu/automatenb/environment/{input.folder_name}"
#         notebook_path = f"{folder_path}/{input.file_name}"
#         result = execute_notebook(notebook_path, folder_path, input.cell_index)
#         if "status" in result and result["status"] == "error":
#             return result

#         # Convert the cell to a .py script
#         exporter = PythonExporter()
#         script, _ = exporter.from_notebook_node(result["notebook"])

#         # Write the script to a .py file
#         script_path = f"{folder_path}/cell_{input.cell_index}.py"
#         print(script_path)
#         with open(script_path, 'w') as f:
#             f.write(script)

#         # Execute the .py script and capture the output
#         process = subprocess.run(["python3", script_path], capture_output=True, text=True)

#         # Delete the .py script after execution
#         os.remove(script_path)

#         if process.returncode != 0:
#             return {"status": "error", "message": process.stderr}
#         else:
#             return {"status": "success", "output": process.stdout}
#     except Exception as e:
#         logging.error(f"Error executing cell as script: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/replace-notebook")
# async def write_notebook(input: ReplaceNotebookData, token: str = Depends(verify_token)):
#     notebook_path = f"/home/ubuntu/automatenb/environment/{input.folder_name}/{input.file_name}"
#     if check_if_file_exists(notebook_path):
#         try:
#             # Convert the dictionary to a notebook object
#             nb = from_dict(input.content)

#             # Write the notebook node to a file
#             with open(notebook_path, 'w') as f:
#                 write(nb, f)

#             return {"status": "success"}
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))
#     else:
#         return {"status": "error", "message": "File does not exist"}

@app.post("/new-notebook/")
async def notebook_environment(notebook: Notebook, token: str = Depends(verify_token)):
    notebook_data = handle_input(notebook)
    folder_name = notebook_data.folder_name
    file_name = notebook_data.file_name
    notebook_content = notebook_data.notebook_data.content.dict()  # Convert to dictionary

    folder_name = f"/home/ubuntu/automatenb/environment/{folder_name}"
    print(folder_name)
    os.makedirs(folder_name, exist_ok=True)

    notebook_path = os.path.join(folder_name, file_name)

    # if check_if_file_exists(notebook_path):
    #     return {"status": "error", "message": "File already exists"}

    try:
        # Convert the list of strings into a single string for each cell
        for cell in notebook_content['cells']:
            cell['source'] = '\n'.join(cell['source'])
        # Convert the dictionary to a notebook object
        nb = from_dict(notebook_content)

        # Write the notebook node to a file
        with open(notebook_path, 'w') as f:
            write(nb, f)
        # Execute the notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': folder_name}})

        # Convert the executed notebook to HTML
        exporter = NotebookExporter()
        (body, resources) = exporter.from_notebook_node(nb)

        return {"status": "success", "output": body}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete-file")
async def delete_file(input: DeleteFileInput, token: str = Depends(verify_token)):
    file_path = f"/home/ubuntu/automatenb/environment/{input.folder_name}/{input.file_name}"
    print(file_path)
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
            return {"status": "success", "message": f"File {input.file_name} has been deleted."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        return {"status": "error", "message": "File does not exist"}

@app.post("/upload-file/{folder_name}")
async def upload_file(folder_name: str, file: UploadFile = File(...), token: str = Depends(verify_token)):
    folder_name = f"/home/ubuntu/automatenb/environment/{folder_name}"
    if not os.path.isdir(folder_name):
        raise HTTPException(status_code=400, detail="Folder does not exist")

    with open(f"{folder_name}/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename}

#Uploads file to fs from any URL
@app.post("/upload-url")
async def download_file(input: DownloadFileInput, token: str = Depends(verify_token)):
    folder_name = f"/home/ubuntu/automatenb/environment/{input.folder_name}"
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name, exist_ok=True)
    file_path = f"{folder_name}/{input.file_name}"

    try:
        with requests.get(input.url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return {"filename": input.file_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete-folder")
async def delete_folder(folder_name: FolderName, token: str = Depends(verify_token)):
    folder_path = f"/home/ubuntu/automatenb/environment/{folder_name.folder_name}"
    if os.path.isdir(folder_path):
        try:
            shutil.rmtree(folder_path)
            return {"status": "success", "message": f"Folder {folder_name.folder_name} has been deleted."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        return {"status": "error", "message": "Folder does not exist"}

@app.post("/add-cell")
async def add_cell(folder_name: str = Body(...), file_name: str = Body(...), cell_content: str = Body(...), cell_type: str = Body(...), token: str = Depends(verify_token)):
    try:
        notebook_path = f"/home/ubuntu/automatenb/environment/{folder_name}/{file_name}"
        with open(notebook_path) as f:
            nb = read(f, as_version=4)

        if cell_type == 'code':
            new_cell = new_code_cell(source=cell_content)
        elif cell_type == 'markdown':
            new_cell = new_markdown_cell(source=cell_content)
        else:
            return {"status": "error", "message": "Invalid cell type"}

        nb.cells.append(new_cell)
        new_cell_index = len(nb.cells)

        with open(notebook_path, 'w') as f:
            write(nb, f)

        return {"status": "success", "cell_index": new_cell_index}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-cell")
async def update_cell(input: UpdateCellInput, token: str = Depends(verify_token)):
    try:
        notebook_path = f"/home/ubuntu/automatenb/environment/{input.folder_name}/{input.file_name}"
        with open(notebook_path) as f:
            nb = read(f, as_version=4)

        if input.cell_type == 'code':
            new_cell = new_code_cell(source=input.cell_content)
        elif input.cell_type == 'markdown':
            new_cell = new_markdown_cell(source=input.cell_content)
        else:
            return {"status": "error", "message": "Invalid cell type"}

        if input.cell_index < 0 or input.cell_index >= len(nb.cells):
            return {"status": "error", "message": "Invalid cell index"}

        nb.cells[input.cell_index] = new_cell

        with open(notebook_path, 'w') as f:
            write(nb, f)

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete-cell")
async def delete_cell(input: DeleteCellInput, token: str = Depends(verify_token)):
    try:
        notebook_path = f"/home/ubuntu/automatenb/environment/{input.folder_name}/{input.file_name}"
        with open(notebook_path) as f:
            nb = read(f, as_version=4)

        if input.cell_index < 0 or input.cell_index >= len(nb.cells):
            return {"status": "error", "message": "Invalid cell index"}

        del nb.cells[input.cell_index]

        with open(notebook_path, 'w') as f:
            write(nb, f)

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# downloads file from fs to client
@app.post("/download-file")
async def download_file(token: str = Depends(verify_token), folder_name: str = Body(...), file_name: str = Body(...)):
    file_path = f"/home/ubuntu/automatenb/environment/{folder_name}/{file_name}"
    if os.path.isfile(file_path):
        return FileResponse(file_path, filename=file_name)
    else:
        raise HTTPException(status_code=404, detail="File not found")


#uploads file to supabase and returns SignedURL  
@app.post("/download-url")
async def upload_file(input: ULFileData, token: str = Depends(verify_token)):
    folder_path = f"/home/ubuntu/automatenb/environment/{input.folder_name}"
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail="Folder does not exist")

    path_on_supabase = f"{input.folder_name}/{input.file_name}"

    file_path = f"{folder_path}/{input.file_name}"
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=400, detail="File does not exist")

    try:

        # Delete the existing file if it exists
        supabase.storage.from_(input.bucket_name).remove([path_on_supabase])

        with open(file_path, 'rb') as f:
            supabase.storage.from_(input.bucket_name).upload(file=f, path=path_on_supabase)

        # Get the file URL
        #file_url = supabase.storage.from_(input.bucket_name).get_public_url(path_on_supabase)
        res = supabase.storage.from_(input.bucket_name).create_signed_url(path_on_supabase, input.expiry_duration)

        #return {"status": "success", "message": f"File {input.file_name} has been uploaded to {input.bucket_name}", "file_url": file_url, "signed_url": res}
        return {"status": "success", "message": f"File {input.file_name} has been uploaded to {input.bucket_name}", "signed_url": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/environment")
async def get_file(token: str = Depends(verify_token)):
    return FileResponse('/home/ubuntu/automatenb/environment.txt')



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
