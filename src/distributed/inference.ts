import { sendChunks } from "./utils.ts";

const ws = new WebSocket('wss://private-inference.onrender.com');
// const ws = new WebSocket('ws://localhost:8080');

ws.onopen = async () => {
    console.log('Connected to server');
//   ws.send('Hello from the client!');

    await encryptImage([
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
    ])

    const encryptedInputs = await Deno.readFile('./keys/encrypted_inputs.bin')

    await sendChunks(encryptedInputs, 'encrypted-inputs', ws)

    await ws.send(JSON.stringify({ id: 'inference' }))
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

async function encryptImage(image: number[]) {
    const command = new Deno.Command("cargo", {
        args: ['run', '--release', '--bin', 'encrypt_image', JSON.stringify(image)],
        stdout: "piped",
        // stderr: "null"
    });

    command.spawn()

    const { stdout } = await command.output();

    const output = new TextDecoder().decode(stdout);

    // console.log(output)

    console.log('Image encrypted!')
}