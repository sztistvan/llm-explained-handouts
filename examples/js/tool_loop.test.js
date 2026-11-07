import test from "node:test";
import assert from "node:assert/strict";
import { runAgent, scriptedModel, validateArgs, tools } from "./tool_loop.js";

test("happy path: model requests a tool, gets the result, then answers", () => {
  const { answer, messages } = runAgent(scriptedModel, "weather?");
  assert.equal(answer, "It is 18°C and cloudy in Budapest.");
  assert.deepEqual(messages.map((m) => m.role), ["user", "assistant", "tool"]);
});

test("invalid arguments are rejected and the error is fed back to the model", () => {
  let call = 0;
  const model = (messages) => {
    call++;
    if (call === 1) return { type: "tool_call", name: "add", arguments: { a: "1", b: 2 } }; // a is a string
    const last = messages.at(-1);
    return { type: "final", content: `tool said: ${last.content.error}` };
  };
  const { answer } = runAgent(model, "add");
  assert.match(answer, /argument "a" must be a number/);
});

test("unknown tools produce an error result instead of crashing", () => {
  let call = 0;
  const model = (messages) =>
    call++ === 0
      ? { type: "tool_call", name: "delete_everything", arguments: {} }
      : { type: "final", content: messages.at(-1).content.error };
  assert.equal(runAgent(model, "x").answer, 'unknown tool "delete_everything"');
});

test("the loop is capped when the model never finishes", () => {
  const forever = () => ({ type: "tool_call", name: "add", arguments: { a: 1, b: 2 } });
  const result = runAgent(forever, "x", tools, 3);
  assert.equal(result.answer, null);
  assert.match(result.error, /stopped after 3 steps/);
  assert.equal(result.messages.filter((m) => m.role === "tool").length, 3);
});

test("validateArgs accepts correct arguments", () => {
  assert.equal(validateArgs(tools.add, { a: 1, b: 2 }), null);
});
