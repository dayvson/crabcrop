/**
 * The server's per-axis pixel cap (`MAX_DIMENSION` in the
 * server's `url.rs`): the largest value a size side, a crop
 * or pad box, a region, a border, a radius, or a pixelate
 * block can name. A larger value is a parse error on the
 * server, so the builders, the explainer, and the framework
 * integrations all check against this one constant.
 */
export const MAX_DIMENSION = 16383;
