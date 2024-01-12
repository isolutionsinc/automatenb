import nbformat
from nbconvert.preprocessors import ExecutePreprocessor



def execute_notebook(notebook_path):
    # Load the notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Set up a notebook execution preprocessor
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    # Execute the notebook
    try:
        ep.preprocess(nb)
    except Exception as e:
        print("Error during notebook execution: ", e)
        return None

    return nb

def get_output_from_cell(nb, cell_index):
    """ Get the output from a specific cell """
    cell = nb.cells[cell_index]
    if cell.cell_type == 'code':
        return cell.outputs
    else:
        return None

# Path to your notebook
notebook_path = './environment/test.ipynb'

# Execute the notebook
nb = execute_notebook(notebook_path)

if nb:
    # Specify the index of the cell you want to get output from
    cell_index = 0  # for example, get output from the first cell

    # Get the output from the specified cell
    output = get_output_from_cell(nb, cell_index)
    print("Output of the cell:", output)
else:
    print("Failed to execute the notebook")

