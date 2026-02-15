const ws = new WebSocket('wss://private-inference.onrender.com');

const CHUNK_SIZE = 64 * 1024;

ws.onopen = async () => {
    console.log('Connected to server');
//   ws.send('Hello from the client!');

    generateKeys([
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
    ])

    const serverKey = await Deno.readFile('./keys/server_key.bin')

    await sendChunks(serverKey, 'server-key')

    const encryptedInputs = await Deno.readFile('./keys/encrypted_inputs.bin')

    await sendChunks(encryptedInputs, 'encrypted-inputs')
    
};

ws.onmessage = (event) => {
  console.log('Received:', event.data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from server');
};

async function sendChunks(data: Uint8Array<ArrayBuffer>, id: string) {
    const totalChunks = Math.ceil(data.length / CHUNK_SIZE);

    for (let i = 0; i < totalChunks; i++) {
        const start = i * CHUNK_SIZE;
        const end = Math.min(start + CHUNK_SIZE, data.length);
        const chunk = data.slice(start, end);
        
        // Convert chunk to base64
        const base64Chunk = btoa(String.fromCharCode(...new Uint8Array(chunk)));
        
        const message = JSON.stringify({
            id,
            index: i,
            total: totalChunks,
            data: base64Chunk
        });
        
        ws.send(message);
        
        await new Promise(resolve => setTimeout(resolve, 5));
    }
}

async function generateKeys(image: number[]) {
    const command = new Deno.Command("cargo", {
        args: ['run', '--release', '--bin', 'generate_keys', JSON.stringify(image)],
        stdout: "piped",
        // stderr: "null"
    });

    command.spawn()

    const { stdout } = await command.output();

    const output = new TextDecoder().decode(stdout);

    // console.log(output)

    console.log('Keys generated!')
}