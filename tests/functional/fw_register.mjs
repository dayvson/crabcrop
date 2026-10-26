// Maps @crabcrop/url to the in-repo SDK source, same trick as the
// frameworks' own test registers, so loaders run exactly as shipped.
import { registerHooks } from "node:module";

registerHooks({
  resolve(specifier, context, next) {
    if (specifier === "@crabcrop/url") {
      return {
        url: new URL(
          "../../sdks/typescript/src/index.ts",
          import.meta.url,
        ).href,
        shortCircuit: true,
      };
    }
    return next(specifier, context);
  },
});
