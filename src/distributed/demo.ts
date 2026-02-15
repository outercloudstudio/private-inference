// Simple Deno web server that runs a command on each request
const PORT = 2000;

async function deploy(values: boolean[]) {
  console.log("Deploying with values:", values);
  
  // Use map instead of filter to convert boolean to 1/-1
  const imageValues = values.map(value => value ? 1 : -1);
  
  const command = new Deno.Command("deno", {
    args: ['task', 'inference', JSON.stringify(imageValues)],
    stdout: "inherit",
    stderr: "inherit",
  });
  
  command.spawn();
}

async function handler(req: Request): Promise<Response> {
  console.log(`Received ${req.method} request to ${new URL(req.url).pathname}`);
  
  // Handle CORS preflight (OPTIONS) request FIRST
  if (req.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
      },
    });
  }
  
  // Handle POST request
  try {
    const body = await req.json();
    const values = body.values;
    
    console.log("Received values:", values);
    
    deploy(values);
    
    return new Response(
      `Deploying with ${values.length} values...`,
      {
        status: 200,
        headers: { 
          "Content-Type": "text/plain",
          "Access-Control-Allow-Origin": "*",
        },
      }
    );
  } catch (error: any) {
    console.error("Error parsing request:", error);
    
    return new Response(
      `Error: ${error.message}`,
      {
        status: 400,
        headers: { 
          "Content-Type": "text/plain",
          "Access-Control-Allow-Origin": "*",
        },
      }
    );
  }
}

console.log(`Server running on http://localhost:${PORT}`);
Deno.serve({ port: PORT }, handler);