// Maps the published package name to the in-repo SDK source,
// so the server can be tested exactly as shipped without an
// install step. Used as a Node --import preload when spawning
// the server in tests; real deployments install @crabcrop/url.
import { registerHooks } from "node:module";

registerHooks({
  resolve(specifier, context, next) {
    if (specifier === "@crabcrop/url") {
      return {
        url: new URL("../../typescript/src/index.ts", import.meta.url).href,
        shortCircuit: true,
      };
    }
    return next(specifier, context);
  },
});
