from fastapi import FastAPI, HTTPException, Body, Response, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from nbformat import from_dict, write, read
from nbformat.v4 import new_code_cell, new_markdown_cell
from nbconvert.preprocessors import ExecutePreprocessor

from typing import Dict
import json
from dotenv import load_dotenv
import os
import shutil
import subprocess

load_dotenv()

from supabase import create_client, Client
from starlette.responses import StreamingResponse
import aiohttp
import aiofiles
from fastapi import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST

#url: str = os.environ.get("SUPABASE_URL")
#key: str = os.environ.get("SUPABASE_KEY")

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

@app.post("/list-files/{folder_name}")
async def list_files(folder_name: str):
    try:
        folder_path = f"/home/ubuntu/automatenb/environment/{folder_name}"
        files = os.listdir(folder_path)
        return {"status": "success", "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/read-notebook", response_class=Response)
async def read_notebook(folder_name: str = Body(...), file_name: str = Body(...)):
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

@app.post("/execute-cell/")
#async def execute(notebook_path: str, cell_index: int):
async def execute(folder_name: str, notebook_path: str = Body(...), cell_index: int = Body(...)):
    try:
        notebook_path = f"/home/ubuntu/automatenb/environment/{folder_name}/{notebook_path}"
        nb = execute_notebook(notebook_path)
        output = get_output_from_cell(nb, cell_index)
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/replace-notebook/")
async def write_notebook(notebook_data: Dict, folder_name: str):
    #notebook_path = notebook_data.get('path')
    #notebook_path = os.path.join(folder_name, notebook_data.get('path'))
    notebook_name = notebook_data.get('path')
    notebook_path = f"/home/ubuntu/automatenb/environment/{folder_name}/{notebook_name}"

    if check_if_file_exists(notebook_path):
        return {"status": "error", "message": "File already exists"}
    try:
        #notebook_path = notebook_data.get('path')
        notebook_content = notebook_data.get('content')

        # Convert the dictionary to a notebook object
        nb = from_dict(notebook_content)

        # Write the notebook node to a file
        with open(notebook_path, 'w') as f:
            write(nb, f)

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/new-notebook/")
async def notebook_environment(notebook_data: Dict, folder_name: str = Body(...), file_name: str = Body(...)):
    # Prepend /environment/ to the folder_name
    #folder_name = os.path.join(os.getcwd(), "/environment/", folder_name)
    folder_name = f"/home/ubuntu/automatenb/environment/{folder_name}"
    print(folder_name)
    # Create a new directory with the folder_name as its name
    os.makedirs(folder_name, exist_ok=True)

    # Get the notebook path and content from the request data
    #notebook_path = notebook_data.get('path')
    notebook_content = notebook_data.get('content')

    # Extract the base name from the notebook path
    #notebook_basename = os.path.basename(notebook_path)

    # Append the new folder name to the notebook path
    #notebook_path = os.path.join(folder_name, notebook_basename)
    notebook_path = os.path.join(folder_name, file_name)

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

@app.post("/delete-file/{folder_name}")
async def delete_file(folder_name: str, filename: str = Body(...)):
    filename= json.loads(filename)['filename']
    file_path = f"/home/ubuntu/automatenb/environment/{folder_name}/{filename}"
    #file_path = f"/home/ubuntu/automatenb/{filename}"
    print(file_path)
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
            return {"status": "success", "message": f"File {filename} has been deleted."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        return {"status": "error", "message": "File does not exist"}

@app.post("/upload-file/{folder_name}")
async def upload_file(folder_name: str, file: UploadFile = File(...)):
    folder_name = f"/home/ubuntu/automatenb/environment/{folder_name}"
    if not os.path.isdir(folder_name):
        raise HTTPException(status_code=400, detail="Folder does not exist")

    with open(f"{folder_name}/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename}


@app.post("/upload-supabase/{folder_name}/{bucket_name}/{file_name}")
async def upload_file(folder_name: str, bucket_name: str, file_name: str):
    folder_path = f"/home/ubuntu/automatenb/environment/{folder_name}"
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail="Folder does not exist")
    # Define the path on Supabase storage
    #path_on_supabase = f"{folder_name}/{file_name}"
    path_on_supabase = f"{folder_name}/{file_name}"

    file_path = f"{folder_path}/{file_name}"
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=400, detail="File does not exist")
    print(file_path)
    # Define the curl command
    #curl_command = f'curl -X POST "{url}/storage/v1/object/{bucket_name}/{path_on_supabase}" -H "Authorization: Bearer {key}" -H "Content-Type: application/octet-stream" --data-binary @{file_path}'
    curl_command = f'curl -X POST "{url}/storage/v1/object/{bucket_name}/{path_on_supabase}" -H "Authorization: Bearer {key}" --data-binary @{file_path}'

    try:
        # Execute the curl command
        print(curl_command)
        process = subprocess.run(curl_command, shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
        response = process.stdout

        # Parse the JSON response
        response_json = json.loads(response)

        # Get the file URL
        #file_url = response_json['publicURL']

        #return {"status": "success", "message": f"File {file_name} has been uploaded to {bucket_name}", "file_url": file_url}
        return {"status": "success", "message": f"File {file_name} has been uploaded to {bucket_name}"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete-folder/{folder_name}")
async def delete_folder(folder_name: str):
    folder_path = f"/home/ubuntu/automatenb/environment/{folder_name}"
    #folder_path = f"/home/ubuntu/automatenb/{folder_name}"
    if os.path.isdir(folder_path):
        try:
            shutil.rmtree(folder_path)
            return {"status": "success", "message": f"Folder {folder_name} has been deleted."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        return {"status": "error", "message": "Folder does not exist"}

from nbformat.v4 import new_code_cell, new_markdown_cell

@app.post("/add-cell/{folder_name}")
async def add_cell(folder_name: str, notebook_name: str = Body(...), cell_content: str = Body(...), cell_type: str = Body(...)):
    try:
        notebook_path = f"/home/ubuntu/automatenb/environment/{folder_name}/{notebook_name}"
        with open(notebook_path) as f:
            nb = read(f, as_version=4)

        if cell_type == 'code':
            new_cell = new_code_cell(source=cell_content)
        elif cell_type == 'markdown':
            new_cell = new_markdown_cell(source=cell_content)
        else:
            return {"status": "error", "message": "Invalid cell type"}

        nb.cells.append(new_cell)

        with open(notebook_path, 'w') as f:
            write(nb, f)

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from nbformat.v4 import new_code_cell, new_markdown_cell

@app.post("/update-cell/{folder_name}")
async def update_cell(folder_name: str, notebook_name: str = Body(...), cell_index: int = Body(...), cell_content: str = Body(...), cell_type: str = Body(...)):
    try:
        notebook_path = f"/home/ubuntu/automatenb/environment/{folder_name}/{notebook_name}"
        with open(notebook_path) as f:
            nb = read(f, as_version=4)

        if cell_type == 'code':
            new_cell = new_code_cell(source=cell_content)
        elif cell_type == 'markdown':
            new_cell = new_markdown_cell(source=cell_content)
        else:
            return {"status": "error", "message": "Invalid cell type"}

        if cell_index < 0 or cell_index >= len(nb.cells):
            return {"status": "error", "message": "Invalid cell index"}

        nb.cells[cell_index] = new_cell

        with open(notebook_path, 'w') as f:
            write(nb, f)

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-file/{folder_name}/{file_name}")
async def download_file(folder_name: str, file_name: str):
    file_path = os.path.join(folder_name, file_name)
    if os.path.isfile(file_path):
        return FileResponse(file_path, filename=file_name)
    else:
        raise HTTPException(status_code=404, detail="File not found")
import aiofiles

@app.get("/download-supabase/{bucket_name}/{folder_name}/{file_name}")
async def download_file(bucket_name: str, folder_name: str, file_name: str):
    # Construct the path on Supabase storage
    path_on_supabase = f"{folder_name}/{file_name}"

    # Download the file from Supabase
    url = supabase.storage.from_(bucket_name).get_public_url(path_on_supabase)

    # Create an aiohttp client session
    async with aiohttp.ClientSession() as session:
        # Make a GET request to the file URL
        async with session.get(url) as resp:
            # Read the file content
            file_content = await resp.read()

    # Define the path where the file will be saved
    save_path = f"/home/ubuntu/automatenb/environment/{folder_name}/{file_name}"

    # Write the file content to a file
    async with aiofiles.open(save_path, 'wb') as f:
        await f.write(file_content)

    return {"status": "success", "message": f"File {file_name} has been downloaded and saved to {folder_name}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
