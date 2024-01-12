from fastapi import FastAPI, HTTPException
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

app = FastAPI()

def execute_notebook(notebook_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

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

@app.post("/execute-notebook/")
async def execute(notebook_path: str, cell_index: int):
    try:
        nb = execute_notebook(notebook_path)
        output = get_output_from_cell(nb, cell_index)
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
