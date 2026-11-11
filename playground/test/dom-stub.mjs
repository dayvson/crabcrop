/**
 * A minimal DOM for driving the page's own app.mjs under Node: the
 * elements app.mjs looks up by id, with only the properties and
 * methods it uses. installDom() sets the globals app.mjs reads and
 * returns the elements by id.
 */
export class StubElement {
  constructor(tag = "div") {
    this.tagName = tag;
    this.children = [];
    this.listeners = {};
    this.value = "";
    this.textContent = "";
    this.className = "";
  }
  addEventListener(name, fn) {
    (this.listeners[name] ??= []).push(fn);
  }
  async dispatch(name) {
    for (const fn of this.listeners[name] ?? []) await fn();
  }
  appendChild(child) {
    this.children.push(child);
    return child;
  }
  append(...children) {
    this.children.push(...children);
  }
  replaceChildren(...children) {
    this.children = children;
  }
  set src(v) {
    this._src = v;
  }
  get src() {
    return this._src;
  }
  removeAttribute(name) {
    if (name === "src") this._src = undefined;
  }
}

const IDS = [
  "base", "secret", "source", "operations", "preset", "expiry",
  "built", "preview", "sizes", "chips", "copy",
  "explain-input", "explain-built", "explanation", "verify-line",
];

export function installDom() {
  const elements = Object.fromEntries(IDS.map((id) => [id, new StubElement()]));
  globalThis.document = {
    getElementById: (id) => elements[id],
    createElement: (tag) => new StubElement(tag),
  };
  Object.defineProperty(globalThis, "navigator", {
    value: { clipboard: { writeText: async () => {} } },
    configurable: true,
  });
  return elements;
}
