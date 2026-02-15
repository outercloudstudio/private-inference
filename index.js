const WebSocket = require('ws');

const PORT = process.env.PORT || 8080;

const wss = new WebSocket.Server({ port: PORT });

console.log('WebSocket server is running on ws://localhost:8080');

let layer = 0
let node = 0
let nodesLeft = 0

let inferenceSocket = undefined

wss.on('connection', (ws) => {
  console.log('New client connected');

  ws.on('message', async (message) => {
    const data = JSON.parse(message)

    console.log(`Received: ${data.id}`);

    if(data.id === 'inference') {
        inferenceSocket = ws

        layer = 0
        node = 0
        nodesLeft = 32

        for(const client of wss.clients) {
            if(client === ws) continue

            if (client.readyState !== WebSocket.OPEN) continue
    
            client.send(JSON.stringify({
                id: 'calculate',
                location: {
                    layer,
                    node
                }
            }));

            node++
        }
    } else if(data.id === 'calculate-finished') {
        nodesLeft--

        console.log(nodesLeft, layer, node)

        if(layer === 0 && node < 32) {
            ws.send(JSON.stringify({
                id: 'calculate',
                location: {
                    layer,
                    node
                }
            }))

            node++
        } else if(layer === 1 && node < 32) {
            ws.send(JSON.stringify({
                id: 'calculate',
                location: {
                    layer,
                    node
                }
            }))

            node++
        } else if(layer === 2 && node < 10) {
            ws.send(JSON.stringify({
                id: 'calculate',
                location: {
                    layer,
                    node
                }
            }))

            node++
        }

        if(nodesLeft === 0 && layer === 0) {
            node = 0
            layer = 1
            nodesLeft = 32

            console.log('Beginning layer 1!')

            for(const client of wss.clients) {
                if(client === inferenceSocket) continue

                if (client.readyState !== WebSocket.OPEN) continue
        
                client.send(JSON.stringify({
                    id: 'calculate',
                    location: {
                        layer,
                        node
                    }
                }));

                node++
            }
        } else if(nodesLeft === 0 && layer === 1) {
            node = 0
            layer = 2
            nodesLeft = 10

            console.log('Beginning layer 2!')

            for(const client of wss.clients) {
                if(client === inferenceSocket) continue
                
                if (client.readyState !== WebSocket.OPEN) continue
        
                client.send(JSON.stringify({
                    id: 'calculate',
                    location: {
                        layer,
                        node
                    }
                }));

                node++
            }
        } else if(nodesLeft === 0 && layer === 2) {
            for(const client of wss.clients) {
                if(client === inferenceSocket) continue

                if (client.readyState !== WebSocket.OPEN) continue

                client.send(JSON.stringify({
                    id: 'inference-complete',
                }));
            }
        }
    } else if(data.id === 'server-key') {
        for(const client of wss.clients) {
            if(client === ws) continue

            if (client.readyState !== WebSocket.OPEN) continue
    
            client.send(JSON.stringify(data));
        }
    } else if(data.id === 'encrypted-zero') {
        for(const client of wss.clients) {
            if(client === ws) continue

            if (client.readyState !== WebSocket.OPEN) continue
    
            client.send(JSON.stringify(data));
        }
    } else if(data.id === 'encrypted-inputs') {
        for(const client of wss.clients) {
            if(client === ws) continue

            if (client.readyState !== WebSocket.OPEN) continue
    
            client.send(JSON.stringify(data));
        }
    } else if(data.id === 'calculate-result') {
        for(const client of wss.clients) {
            if(client === ws) continue

            if (client.readyState !== WebSocket.OPEN) continue
    
            client.send(JSON.stringify(data));
        }
    }
  });

  ws.on('close', () => {
    console.log('Client disconnected');
  });

  ws.on('error', (error) => {
    console.error('WebSocket error:', error);
  });
});

wss.on('error', (error) => {
  console.error('Server error:', error);
});