import { sendChunks } from "./utils.ts";

const ws = new WebSocket('wss://private-inference.onrender.com');

ws.onopen = async () => {
    console.log('Connected to server');
};

ws.onmessage = async (event) => {
    const message = JSON.parse(event.data)

    console.log('Received:', message.id);

    if(message.id === 'calculate') {
        const result = await calculate(message.location)

        await sendChunks(result, 'calculate-result', ws)

        await ws.send(JSON.stringify({
            id: 'calculate-finished',
        }))
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
        stdout: "piped",
        // stderr: "null"
    });

    command.spawn()

    const { stdout } = await command.output();

    const output = new TextDecoder().decode(stdout);

    const result = await Deno.readFile(`./keys/layer_${location.layer}_${location.node}.bin`)

    console.log('Calculated!', location)

    return result
}