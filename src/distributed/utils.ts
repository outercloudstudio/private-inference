const CHUNK_SIZE = 64 * 1024;

export async function sendChunks(data: Uint8Array<ArrayBuffer>, id: string, ws: WebSocket, extraData?: any) {
    const totalChunks = Math.ceil(data.length / CHUNK_SIZE);

    for (let i = 0; i < totalChunks; i++) {
        const start = i * CHUNK_SIZE;
        const end = Math.min(start + CHUNK_SIZE, data.length);
        const chunk = data.slice(start, end);
        
        // Convert chunk to base64
        const base64Chunk = btoa(String.fromCharCode(...new Uint8Array(chunk)));
        
        if(extraData) {
            const message = JSON.stringify({
                id,
                index: i,
                total: totalChunks,
                data: base64Chunk,
                ...extraData
            });
            
            ws.send(message);
        } else {
            const message = JSON.stringify({
                id,
                index: i,
                total: totalChunks,
                data: base64Chunk,
            });
            
            ws.send(message);
        }
        
        await new Promise(resolve => setTimeout(resolve, 5));
    }
}