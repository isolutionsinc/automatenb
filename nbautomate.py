import nbformat
from nbclient import NotebookClient

notebook_filename = '/home/ubuntu/automatenb/environment/test.ipynb'


nb = nbformat.read(notebook_filename, as_version=4)
client = NotebookClient(nb, timeout=600, kernel_name='python3')

client.execute()

nbformat.write(nb, 'executed_notebook.ipynb')


