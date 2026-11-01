// The example app's shared crabcrop loader. The "card"
// preset carries the whole transform policy on the server;
// components name no operations, only widths. Against a
// local crabcrop (the repository README's quick start) the
// base is the local origin; preset mode pairs with the
// server running CRABCROP_ALLOW_UNSAFE=1.
import { createCrabcropLoader } from "./crabcrop-loader";

export const images = createCrabcropLoader({
  base: "http://localhost:3000",
  preset: "card",
});
