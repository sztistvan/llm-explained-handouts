// A minimal tool-use (function calling) loop (handout 08).
//
// The LLM never runs anything. It only EMITS a structured request ("call tool X with args Y").
// Your code validates it, runs it, appends the result to the conversation, and calls the model again.

export const tools = {
  get_weather: {
    description: "Get the current weather for a city",
    parameters: { city: "string" },
    run: ({ city }) => ({ city, tempC: 18, sky: "cloudy" }), // stand-in for a real API call
  },
  add: {
    description: "Add two numbers",
    parameters: { a: "number", b: "number" },
    run: ({ a, b }) => ({ sum: a + b }),
  },
};

// Never trust model-produced arguments: validate before executing.
export function validateArgs(tool, args) {
  for (const [name, type] of Object.entries(tool.parameters)) {
    if (typeof args?.[name] !== type) return `argument "${name}" must be a ${type}`;
  }
  return null;
}

/**
 * modelFn(messages) -> { type: "tool_call", name, arguments } | { type: "final", content }
 * (A real model API returns an equivalent structure, in a vendor-specific format.)
 */
export function runAgent(modelFn, userMessage, toolset = tools, maxSteps = 5) {
  const messages = [{ role: "user", content: userMessage }];
  for (let step = 0; step < maxSteps; step++) {
    const reply = modelFn(messages);
    if (reply.type === "final") return { answer: reply.content, messages };

    messages.push({ role: "assistant", toolCall: reply });
    const tool = toolset[reply.name];
    let result;
    if (!tool) result = { error: `unknown tool "${reply.name}"` };
    else {
      const problem = validateArgs(tool, reply.arguments);
      result = problem ? { error: problem } : tool.run(reply.arguments); // errors go back to the model
    }
    messages.push({ role: "tool", name: reply.name, content: result });
  }
  return { answer: null, error: `stopped after ${maxSteps} steps`, messages }; // always cap the loop
}

// Scripted stand-in for an LLM so the example runs offline.
export function scriptedModel(messages) {
  const lastTool = [...messages].reverse().find((m) => m.role === "tool");
  if (!lastTool) return { type: "tool_call", name: "get_weather", arguments: { city: "Budapest" } };
  const { city, tempC, sky } = lastTool.content;
  return { type: "final", content: `It is ${tempC}°C and ${sky} in ${city}.` };
}

if (process.argv[1]?.endsWith("tool_loop.js")) { // run the demo only when executed directly
  const { answer, messages } = runAgent(scriptedModel, "What's the weather in Budapest?");
  console.log("Conversation:");
  for (const m of messages) console.log(" ", m.role, JSON.stringify(m.toolCall ?? m.content));
  console.log("Answer:", answer);
}
