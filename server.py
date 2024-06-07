import socket
import pickle
from typing import List, Optional
from llama import Llama
from time import time

def handle_client(client_socket, model):
    # Receive data from the client
    data = []
    # start = time()
    while True:
        packet = client_socket.recv(4096)
        if packet is None or packet == b'': #time() - start > 2:
            break
        data.append(packet)
    data = b''.join(data)
    
    if data:
        # Deserialize the data
        request = pickle.loads(data)
        dialogs = request.get('dialogs')
        temperature = request.get('temperature', 0.6)
        max_gen_len = request.get('max_gen_len')
        action_list = request.get('action_list', ["scoop", "fork", "cut", "move", "stir", "DONE"])

        print(f"Received data: {dialogs}")
        print(f"Temperature: {temperature}")
        print(f"Max Gen Len: {max_gen_len}")
        print(f"Action List: {action_list}")
        # Call the model's get_sementic_score method
        response = model.get_semantic_score(dialogs, temperature, max_gen_len, action_list)
        

        # Serialize the response
        response_data = pickle.dumps(response)
        

        # Send the response back to the client
        client_socket.send(response_data)
    client_socket.close()

def main():
    model = Llama.build(
        ckpt_dir="Meta-Llama-3-8B-Instruct/",
        tokenizer_path="Meta-Llama-3-8B-Instruct/tokenizer.model",
        max_seq_len=2048,
        max_batch_size=4,
    )
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server.bind(('0.0.0.0', 9999))
    server.listen(5)
    print("Server listening on port 9999")

    while True:
        client_socket, addr = server.accept()
        print(f"Accepted connection from {addr}")
        handle_client(client_socket, model)

if __name__ == '__main__':
    main()