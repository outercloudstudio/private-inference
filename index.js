const WebSocket = require('ws');

const PORT = process.env.PORT || 8080;

const wss = new WebSocket.Server({ port: PORT });

console.log('WebSocket server is running on ws://localhost:8080');

wss.on('connection', (ws) => {
  console.log('New client connected');

  let layer = 0
  let node = 0

  ws.on('message', (message) => {
    const data = JSON.parse(message)

    console.log(`Received: ${data.id}`);

    if(data.id === 'inference') {
        layer = 0
        node = 0

        for(const client of wss.clients) {
            if (client.readyState !== WebSocket.OPEN) continue
    
            client.send(JSON.stringify({
                id: 'calculate',
                location: {
                    layer,
                    node
                }
            }));
        }
    } else if(data.id === 'calculate-finished') {
        node++

        if(node < 64) {
            ws.send(JSON.stringify({
                id: 'calculate',
                location: {
                    layer,
                    node
                }
            }))
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