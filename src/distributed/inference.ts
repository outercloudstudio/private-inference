import { sendChunks } from "./utils.ts";

// const ws = new WebSocket('wss://private-inference.onrender.com');
const ws = new WebSocket('ws://localhost:8080');

let resultBuffers: Record<string, Uint8Array> = {}

ws.onopen = async () => {
    console.log('Connected to server');

    await encryptImage([
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1,  1, -1, -1, -1,
        -1, -1, -1,  1, -1, -1, -1,
        -1, -1, -1,  1, -1, -1, -1,
        -1, -1, -1,  1, -1, -1, -1,
        -1, -1, -1,  1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
    ])

    const encryptedInputs = await Deno.readFile('./keys/encrypted_inputs.bin')

    await sendChunks(encryptedInputs, 'encrypted-inputs', ws)

    await ws.send(JSON.stringify({ id: 'inference' }))
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data)

    console.log('Received:', message.id);

    if(message.id === 'calculate-result') {
        console.log(message.index, message.total, message.location)

        const key = `${message.location.layer}_${message.location.node}`

        if(message.index === 0) resultBuffers[key] = new Uint8Array()

        const chunk = Uint8Array.from(atob(message.data), c => c.charCodeAt(0));

        const mergedBuffer = new Uint8Array(resultBuffers[key].length + chunk.length);
        mergedBuffer.set(resultBuffers[key]!)
        mergedBuffer.set(chunk, resultBuffers[key].length)

        resultBuffers[key] = mergedBuffer

        if(message.index === message.total - 1) {
            Deno.writeFileSync(`./keys/layer_${message.location.layer}_${message.location.node}.bin`, mergedBuffer)

            delete resultBuffers[key]
        }
    }
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