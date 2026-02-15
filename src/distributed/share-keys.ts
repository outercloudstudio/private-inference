import { sendChunks } from "./utils.ts";

const ws = new WebSocket('wss://private-inference.onrender.com');
// const ws = new WebSocket('ws://localhost:8080');

ws.onopen = async () => {
    console.log('Connected to server');
//   ws.send('Hello from the client!');

    await generateKeys()

    const serverKey = await Deno.readFile('./keys/server_key.bin')

    await sendChunks(serverKey, 'server-key', ws)

    ws.close()
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

async function generateKeys() {
    const command = new Deno.Command("cargo", {
        args: ['run', '--release', '--bin', 'generate_keys'],
        stdout: "piped",
        // stderr: "null"
    });

    command.spawn()

    const { stdout } = await command.output();

    const output = new TextDecoder().decode(stdout);

    // console.log(output)

    console.log('Keys generated!')
}