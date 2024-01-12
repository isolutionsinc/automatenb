import json
import requests
import datetime
import uuid
from pprint import pprint
from websocket import create_connection

# The token is written on stdout when you start the notebook
notebook_path = '/environment/test.ipynb'
base = 'http://localhost:8888'
headers = {'Authorization': 'Token 2b787c0cfcac2defe275a8d4e2ba68e4845fc57a03dca244'}

url = base + '/api/kernels'
response = requests.post(url,headers=headers)
kernel = json.loads(response.text)

# Load the notebook and get the code of each cell
url = base + '/api/contents' + notebook_path
response = requests.get(url,headers=headers)
file = json.loads(response.text)
code = [ c['source'] for c in file['content']['cells'] if len(c['source'])>0 ]

# Execution request/reply is done on websockets channels
ws = create_connection("ws://localhost:8888/api/kernels/"+kernel["id"]+"/channels",
     header=headers)

def send_execute_request(code):
    msg_type = 'execute_request';
    content = { 'code' : code, 'silent':False }
    hdr = { 'msg_id' : uuid.uuid1().hex, 
        'username': 'test', 
        'session': uuid.uuid1().hex, 
        'data': datetime.datetime.now().isoformat(),
        'msg_type': msg_type,
        'version' : '5.0' }
    msg = { 'header': hdr, 'parent_header': hdr, 
        'metadata': {},
        'content': content }
    return msg

for c in code:
    ws.send(json.dumps(send_execute_request(c)))

# We ignore all the other messages, we just get the code execution output
# (this needs to be improved for production to take into account errors, large cell output, images, etc.)
for i in range(0, len(code)):
    msg_type = '';
    while msg_type != "stream":
        rsp = json.loads(ws.recv())
        msg_type = rsp["msg_type"]
    print(rsp["content"]["text"])

ws.close()

send_execute_request("print('Hello World!')")