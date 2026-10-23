// Maps the published package name to the in-repo SDK source,
// so the loaders can be tested exactly as shipped (bare
// `@crabcrop/url` import included) without an install step.
import { registerHooks } from "node:module";

registerHooks({
  resolve(specifier, context, next) {
    if (specifier === "@crabcrop/url") {
      return {
        url: new URL("../../../sdks/typescript/src/index.ts", import.meta.url).href,
        shortCircuit: true,
      };
    }
    return next(specifier, context);
  },
});
