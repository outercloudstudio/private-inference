import { sendChunks } from "../../src/distributed/utils.ts";

// const ws = new WebSocket('wss://private-inference.onrender.com');
const ws = new WebSocket('ws://localhost:8080');

ws.onopen = async () => {
    console.log('Connected to server');
};

let downloadBuffer = undefined

let resultBuffers: Record<string, Uint8Array> = {}

ws.onmessage = async (event) => {
    const message = JSON.parse(event.data)

    console.log('Received:', message.id);

    if(message.id === 'calculate') {
        const result = await calculate(message.location)

        await sendChunks(result, 'calculate-result', ws, {
            location: message.location,
        })

        await ws.send(JSON.stringify({
            id: 'calculate-finished',
        }))
    } else if(message.id === 'server-key') {
        console.log(message.index, message.total)

        if(message.index === 0) downloadBuffer = new Uint8Array()

        const chunk = Uint8Array.from(atob(message.data), c => c.charCodeAt(0));

        const mergedBuffer = new Uint8Array(downloadBuffer.length + chunk.length);
        mergedBuffer.set(downloadBuffer!)
        mergedBuffer.set(chunk, downloadBuffer.length)

        downloadBuffer = mergedBuffer

        if(message.index === message.total - 1) Deno.writeFileSync('./keys/server_key.bin', downloadBuffer)
    } else if(message.id === 'encrypted-zero') {
        console.log(message.index, message.total)

        if(message.index === 0) downloadBuffer = new Uint8Array()

        const chunk = Uint8Array.from(atob(message.data), c => c.charCodeAt(0));

        const mergedBuffer = new Uint8Array(downloadBuffer.length + chunk.length);
        mergedBuffer.set(downloadBuffer!)
        mergedBuffer.set(chunk, downloadBuffer.length)

        downloadBuffer = mergedBuffer

        if(message.index === message.total - 1) Deno.writeFileSync('./keys/encrypted_zero.bin', downloadBuffer)
    } else if(message.id === 'encrypted-inputs') {
        console.log(message.index, message.total)

        if(message.index === 0) downloadBuffer = new Uint8Array()

        const chunk = Uint8Array.from(atob(message.data), c => c.charCodeAt(0));

        const mergedBuffer = new Uint8Array(downloadBuffer.length + chunk.length);
        mergedBuffer.set(downloadBuffer!)
        mergedBuffer.set(chunk, downloadBuffer.length)

        downloadBuffer = mergedBuffer

        if(message.index === message.total - 1) Deno.writeFileSync('./keys/encrypted_inputs.bin', downloadBuffer)
    } else if(message.id === 'calculate-result') {
        console.log(message.index, message.total, message.instance, message.location)

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

async function calculate(location: { node: number, layer: number }) {
    console.log('Calculateding...', location)
    
    const command = new Deno.Command("cargo", {
        args: ['run', '--release', '--bin', 'calculate', JSON.stringify(location)],
        // stdout: "piped",
        // stderr: "null"
    });

    const child = command.spawn()

    const status = await child.status;

    // const { stdout } = await command.output();

    // const output = new TextDecoder().decode(stdout);

    const result = await Deno.readFile(`./keys/layer_${location.layer}_${location.node}.bin`)

    console.log('Calculated!', location)

    return result
}

async function fileExists(path: string): Promise<boolean> {
  try {
    await Deno.stat(path);
    return true;
  } catch (error) {
    if (error instanceof Deno.errors.NotFound) {
      return false;
    }
    throw error; // Re-throw other errors
  }
}