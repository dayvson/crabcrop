// The example app's shared crabcrop loader. The "card"
// preset carries the whole transform policy on the server;
// components name no operations, only widths. Against a
// local crabcrop (the repository README's quick start) the
// base is the local origin, and the unsigned URLs need the
// server running CRABCROP_ALLOW_UNSAFE=1: a local development
// setup only, since unsafe mode alone is an open image proxy.
// Production signs in server code (see the README).
import { createCrabcropLoader } from "./crabcrop-loader";

export const images = createCrabcropLoader({
  base: "http://localhost:3000",
  preset: "card",
});
