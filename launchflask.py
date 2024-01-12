from flask import Flask, request, jsonify
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os


file_path = './environment/test.py'
if os.path.isfile(file_path):
    print("File exists")
else:
    print("File does not exist")

app = Flask(__name__)

notebook_path = './environment/test.py'

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

@app.route('/execute_notebook', methods=['POST'])
def execute_notebook_route():
    data = request.get_json()
    notebook_path = data.get('notebook_path')
    nb = execute_notebook(notebook_path)
    if nb:
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'failure'})

if __name__ == '__main__':
    app.run(debug=True)