<?php
/**
 * Plugin Name:       Crabcrop
 * Plugin URI:        https://crabcrop.org
 * Description:       Serve this site's media through a self-hosted crabcrop instance. Rewrites media URLs (featured images, post content, and srcset) to crabcrop URLs; the media library itself is untouched, and unmapped sizes and any misconfiguration fall back to the original URLs.
 * Version:           0.9.6
 * Author:            crabcrop contributors
 * License:           MIT
 * License URI:       https://opensource.org/licenses/MIT
 * Text Domain:       crabcrop
 */

if (!defined('ABSPATH')) {
    exit;
}

require_once plugin_dir_path(__FILE__) . 'includes/class-crabcrop-url.php';

/*
 * Configuration.
 *
 * Every setting can come from three places. In order of precedence,
 * highest first:
 *
 *   1. Constants in wp-config.php: CRABCROP_WP_BASE_URL,
 *      CRABCROP_WP_MODE, CRABCROP_WP_SECRET, CRABCROP_WP_KEY_ID,
 *      CRABCROP_WP_SIZE_MAP, CRABCROP_WP_DISABLE_SIZES.
 *   2. Filters: crabcrop_wp_base_url, crabcrop_wp_mode,
 *      crabcrop_wp_secret, crabcrop_wp_key_id, crabcrop_wp_size_map,
 *      crabcrop_wp_disable_sizes.
 *   3. The settings screen (Settings > Crabcrop).
 *
 * Code-level configuration wins over the admin screen on purpose: a
 * deployment managed in wp-config.php or a must-use plugin cannot be
 * silently repointed from the dashboard.
 */

function crabcrop_wp_config()
{
    if (isset($GLOBALS['crabcrop_wp_config_memo'])) {
        return $GLOBALS['crabcrop_wp_config_memo'];
    }

    $options = get_option('crabcrop_settings', array());
    if (!is_array($options)) {
        $options = array();
    }
    $config = array(
        'base_url' => isset($options['base_url']) ? (string) $options['base_url'] : '',
        'mode' => isset($options['mode']) ? (string) $options['mode'] : 'presets',
        'secret' => isset($options['secret']) ? (string) $options['secret'] : '',
        'key_id' => '',
        'size_map' => isset($options['size_map']) && is_array($options['size_map'])
            ? $options['size_map'] : array(),
        'disable_sizes' => !empty($options['disable_sizes']),
    );

    if (has_filter('crabcrop_wp_base_url')) {
        $config['base_url'] = (string) apply_filters('crabcrop_wp_base_url', $config['base_url']);
    }
    if (has_filter('crabcrop_wp_mode')) {
        $config['mode'] = (string) apply_filters('crabcrop_wp_mode', $config['mode']);
    }
    if (has_filter('crabcrop_wp_secret')) {
        $config['secret'] = (string) apply_filters('crabcrop_wp_secret', $config['secret']);
    }
    if (has_filter('crabcrop_wp_key_id')) {
        $config['key_id'] = (string) apply_filters('crabcrop_wp_key_id', $config['key_id']);
    }
    if (has_filter('crabcrop_wp_size_map')) {
        $map = apply_filters('crabcrop_wp_size_map', $config['size_map']);
        if (is_array($map)) {
            $config['size_map'] = $map;
        }
    }
    if (has_filter('crabcrop_wp_disable_sizes')) {
        $config['disable_sizes'] = (bool) apply_filters(
            'crabcrop_wp_disable_sizes',
            $config['disable_sizes']
        );
    }

    if (defined('CRABCROP_WP_BASE_URL')) {
        $config['base_url'] = (string) CRABCROP_WP_BASE_URL;
    }
    if (defined('CRABCROP_WP_MODE')) {
        $config['mode'] = (string) CRABCROP_WP_MODE;
    }
    if (defined('CRABCROP_WP_SECRET')) {
        $config['secret'] = (string) CRABCROP_WP_SECRET;
    }
    if (defined('CRABCROP_WP_KEY_ID')) {
        $config['key_id'] = (string) CRABCROP_WP_KEY_ID;
    }
    if (defined('CRABCROP_WP_SIZE_MAP')) {
        $map = CRABCROP_WP_SIZE_MAP;
        if (is_string($map)) {
            $map = crabcrop_wp_parse_size_map($map, $config['mode']);
        }
        if (is_array($map)) {
            $config['size_map'] = $map;
        }
    }
    if (defined('CRABCROP_WP_DISABLE_SIZES')) {
        $config['disable_sizes'] = (bool) CRABCROP_WP_DISABLE_SIZES;
    }

    $config['base_url'] = rtrim($config['base_url'], '/');
    if (!in_array($config['mode'], array('presets', 'signed'), true)) {
        $config['mode'] = 'presets';
    }
    $GLOBALS['crabcrop_wp_config_memo'] = $config;
    return $config;
}

/** Test support: forget the memoized configuration. */
function crabcrop_wp_reset_config()
{
    unset($GLOBALS['crabcrop_wp_config_memo']);
}

/*
 * Size map parsing. The settings screen stores the map as text, one
 * "size = target" line per size; # starts a comment. In presets mode
 * a target is a preset name. In signed mode a target is an operation
 * string used verbatim (for example "crop:300x300:smart" or
 * "preset:card"), so it must be a legal URL remainder fragment: no
 * whitespace, ?, #, or source smuggled in as a URL.
 */

function crabcrop_wp_parse_size_map($text, $mode)
{
    $map = array();
    $lines = preg_split('/\r\n|\r|\n/', (string) $text);
    foreach ($lines as $line) {
        $line = trim($line);
        if ($line === '' || $line[0] === '#') {
            continue;
        }
        $parts = explode('=', $line, 2);
        if (count($parts) !== 2) {
            continue;
        }
        $size = trim($parts[0]);
        $target = trim($parts[1]);
        if (!preg_match('/^[A-Za-z0-9_-]+$/', $size)) {
            continue;
        }
        if ($mode === 'signed') {
            if ($target === ''
                || preg_match('/[\x00-\x20\x7F-\xFF?#]/', $target)
                || strpos($target, '://') !== false) {
                continue;
            }
        } else {
            if (!preg_match('/^[A-Za-z0-9_-]+$/', $target)) {
                continue;
            }
        }
        $map[$size] = $target;
    }
    return $map;
}

function crabcrop_wp_size_map_to_text($map)
{
    $lines = array();
    foreach ((array) $map as $size => $target) {
        $lines[] = $size . ' = ' . $target;
    }
    return implode("\n", $lines);
}

/*
 * URL rewriting.
 *
 * The failure rule everywhere: if anything is off (no base URL, a
 * signed mode without a secret, an unmapped size, a URL that is not
 * this site's own media, or input the builder rejects), the original
 * URL is returned unchanged. A crabcrop outage or a configuration
 * mistake must never break an image on the site.
 */

/** True when $url lives under this site's uploads directory. */
function crabcrop_wp_is_media_url($url)
{
    if (!is_string($url) || $url === '') {
        return false;
    }
    $host = parse_url($url, PHP_URL_HOST);
    if (!is_string($host) || $host === '') {
        return false;
    }
    $uploads = wp_upload_dir();
    if (!empty($uploads['baseurl'])) {
        if (parse_url($uploads['baseurl'], PHP_URL_HOST) === $host) {
            return true;
        }
    }
    return parse_url(get_site_url(), PHP_URL_HOST) === $host;
}

/**
 * Build the crabcrop URL for $source with $operations, or null when
 * the plugin cannot or should not rewrite. $size_name and
 * $attachment_id are context for the crabcrop_wp_url filter.
 */
function crabcrop_wp_build($source, $operations, $size_name, $attachment_id)
{
    $config = crabcrop_wp_config();
    if ($config['base_url'] === '') {
        return null;
    }
    try {
        if ($config['mode'] === 'signed') {
            if ($config['secret'] === '') {
                return null;
            }
            $built = CrabcropUrl::build(
                $config['base_url'],
                $config['secret'],
                $operations,
                $source,
                null,
                $config['key_id'] !== '' ? $config['key_id'] : null
            );
        } else {
            $built = CrabcropUrl::build($config['base_url'], null, $operations, $source);
        }
    } catch (InvalidArgumentException $e) {
        return null;
    }
    return apply_filters('crabcrop_wp_url', $built, $size_name, $attachment_id);
}

/**
 * Build the crabcrop URL for a named WordPress size, through its
 * mapping, or null when the size is unmapped. In presets mode the
 * mapping names a preset; in signed mode the mapping is the
 * operation string itself (a "preset:<name>" value also works).
 * $preferred_source replaces $url as the crabcrop source when given
 * (the original file, so any transform derives from full pixels).
 */
function crabcrop_wp_build_for_size($url, $size_name, $attachment_id, $preferred_source = null)
{
    $config = crabcrop_wp_config();
    if (!isset($config['size_map'][$size_name])) {
        return null;
    }
    $target = $config['size_map'][$size_name];
    $operations = $config['mode'] === 'signed' ? $target : 'preset:' . $target;
    $source = $preferred_source !== null ? $preferred_source : $url;
    return crabcrop_wp_build($source, $operations, $size_name, $attachment_id);
}

/** The attachment's original file URL from its meta, or null. */
function crabcrop_wp_original_url($image_meta)
{
    if (!is_array($image_meta) || empty($image_meta['file'])) {
        return null;
    }
    $uploads = wp_upload_dir();
    if (empty($uploads['baseurl'])) {
        return null;
    }
    return rtrim($uploads['baseurl'], '/') . '/' . $image_meta['file'];
}

/** The registered size name a srcset entry came from, or null. */
function crabcrop_wp_srcset_size_name($source, $image_meta)
{
    if (!is_array($image_meta) || empty($image_meta['sizes']) || !is_array($image_meta['sizes'])) {
        return null;
    }
    if (!isset($source['value']) || !isset($source['url'])) {
        return null;
    }
    foreach ($image_meta['sizes'] as $name => $data) {
        if ((int) $data['width'] === (int) $source['value']
            && basename($source['url']) === $data['file']) {
            return $name;
        }
    }
    return null;
}

function crabcrop_wp_filter_attachment_url($url, $attachment_id)
{
    if (!crabcrop_wp_is_media_url($url)) {
        return $url;
    }
    // The bare attachment URL is rewritten only when the operator
    // mapped the "full" size; everything else keeps the original.
    $built = crabcrop_wp_build_for_size($url, 'full', $attachment_id);
    return $built !== null ? $built : $url;
}
add_filter('wp_get_attachment_url', 'crabcrop_wp_filter_attachment_url', 10, 2);

function crabcrop_wp_filter_attachment_image_src($image, $attachment_id, $size, $icon)
{
    if (!is_array($image) || !isset($image[0]) || !is_string($size)) {
        return $image;
    }
    if (!crabcrop_wp_is_media_url($image[0])) {
        return $image;
    }
    $built = crabcrop_wp_build_for_size($image[0], $size, $attachment_id);
    if ($built !== null) {
        $image[0] = $built;
    }
    return $image;
}
add_filter('wp_get_attachment_image_src', 'crabcrop_wp_filter_attachment_image_src', 10, 4);

function crabcrop_wp_filter_srcset($sources, $size_array, $image_src, $image_meta, $attachment_id)
{
    if (!is_array($sources) || $sources === array()) {
        return $sources;
    }
    $original = crabcrop_wp_original_url($image_meta);
    foreach ($sources as $key => $source) {
        if (!isset($source['url']) || !crabcrop_wp_is_media_url($source['url'])) {
            continue;
        }
        if (isset($source['descriptor']) && $source['descriptor'] !== 'w') {
            continue;
        }
        $built = null;
        $size_name = crabcrop_wp_srcset_size_name($source, $image_meta);
        if ($size_name !== null) {
            $built = crabcrop_wp_build_for_size($source['url'], $size_name, $attachment_id, $original);
        } elseif (crabcrop_wp_config()['mode'] === 'signed'
            && $original !== null
            && isset($source['value']) && (int) $source['value'] > 0) {
            // A width WordPress offers that no registered size owns
            // (typically the full width): in signed mode it still
            // maps to a faithful width-only resize of the original.
            $built = crabcrop_wp_build(
                $original,
                ((int) $source['value']) . 'x0',
                null,
                $attachment_id
            );
        }
        if ($built !== null) {
            $sources[$key]['url'] = $built;
        }
    }
    return $sources;
}
add_filter('wp_calculate_image_srcset', 'crabcrop_wp_filter_srcset', 10, 5);

/*
 * The opt-in sizes toggle: when on, WordPress stops generating
 * intermediate sizes at upload time. The originals are untouched
 * either way; crabcrop produces every displayed size on demand.
 */
function crabcrop_wp_filter_intermediate_sizes($sizes)
{
    return crabcrop_wp_config()['disable_sizes'] ? array() : $sizes;
}
add_filter('intermediate_image_sizes_advanced', 'crabcrop_wp_filter_intermediate_sizes');

/*
 * Settings screen (Settings > Crabcrop).
 */

function crabcrop_wp_sanitize_settings($input)
{
    $out = array(
        'base_url' => '',
        'mode' => 'presets',
        'secret' => '',
        'size_map' => array(),
        'disable_sizes' => false,
    );
    if (!is_array($input)) {
        return $out;
    }
    if (isset($input['base_url'])) {
        $base = rtrim(esc_url_raw(trim((string) $input['base_url'])), '/');
        $scheme = parse_url($base, PHP_URL_SCHEME);
        $out['base_url'] = ($scheme === 'http' || $scheme === 'https') ? $base : '';
    }
    if (isset($input['mode']) && in_array($input['mode'], array('presets', 'signed'), true)) {
        $out['mode'] = $input['mode'];
    }
    if (isset($input['secret'])) {
        $out['secret'] = (string) $input['secret'];
    }
    $out['size_map'] = crabcrop_wp_parse_size_map(
        isset($input['size_map_text']) ? $input['size_map_text'] : '',
        $out['mode']
    );
    $out['disable_sizes'] = !empty($input['disable_sizes']);
    return $out;
}

function crabcrop_wp_constant_for($key)
{
    $names = array(
        'base_url' => 'CRABCROP_WP_BASE_URL',
        'mode' => 'CRABCROP_WP_MODE',
        'secret' => 'CRABCROP_WP_SECRET',
        'size_map' => 'CRABCROP_WP_SIZE_MAP',
        'disable_sizes' => 'CRABCROP_WP_DISABLE_SIZES',
    );
    return isset($names[$key]) && defined($names[$key]) ? $names[$key] : null;
}

function crabcrop_wp_settings_page()
{
    if (!current_user_can('manage_options')) {
        return;
    }
    $options = get_option('crabcrop_settings', array());
    if (!is_array($options)) {
        $options = array();
    }
    $values = array(
        'base_url' => isset($options['base_url']) ? $options['base_url'] : '',
        'mode' => isset($options['mode']) ? $options['mode'] : 'presets',
        'secret' => isset($options['secret']) ? $options['secret'] : '',
        'size_map_text' => crabcrop_wp_size_map_to_text(
            isset($options['size_map']) ? $options['size_map'] : array()
        ),
        'disable_sizes' => !empty($options['disable_sizes']),
    );
    ?>
    <div class="wrap">
      <h1>Crabcrop</h1>
      <p>Serve this site's media through your self-hosted crabcrop instance.
      The media library is untouched: disabling the plugin returns every
      URL to its original form.</p>
      <form method="post" action="options.php">
        <?php settings_fields('crabcrop_settings_group'); ?>
        <table class="form-table" role="presentation">
          <tr>
            <th scope="row"><label for="crabcrop-base-url">Instance base URL</label></th>
            <td>
              <input type="url" id="crabcrop-base-url" class="regular-text"
                name="crabcrop_settings[base_url]"
                value="<?php echo esc_attr($values['base_url']); ?>"
                placeholder="https://images.example.com"
                <?php disabled(crabcrop_wp_constant_for('base_url') !== null); ?>>
              <?php if (crabcrop_wp_constant_for('base_url')) : ?>
                <p class="description">Set by the
                <code><?php echo esc_html(crabcrop_wp_constant_for('base_url')); ?></code>
                constant in wp-config.php.</p>
              <?php else : ?>
                <p class="description">The origin your crabcrop instance answers on.</p>
              <?php endif; ?>
            </td>
          </tr>
          <tr>
            <th scope="row">Mode</th>
            <td>
              <fieldset <?php disabled(crabcrop_wp_constant_for('mode') !== null); ?>>
                <label>
                  <input type="radio" name="crabcrop_settings[mode]" value="presets"
                    <?php checked($values['mode'], 'presets'); ?>>
                  Presets only (recommended): WordPress sizes map to preset
                  names, and no secret lives in WordPress. The instance
                  runs with <code>CRABCROP_ONLY_PRESETS=1</code> and
                  <code>CRABCROP_ALLOW_UNSAFE=1</code>.
                </label><br>
                <label>
                  <input type="radio" name="crabcrop_settings[mode]" value="signed"
                    <?php checked($values['mode'], 'signed'); ?>>
                  Signed: sizes map to operation strings, signed with the
                  secret below.
                </label>
              </fieldset>
              <?php if (crabcrop_wp_constant_for('mode')) : ?>
                <p class="description">Set by the
                <code><?php echo esc_html(crabcrop_wp_constant_for('mode')); ?></code>
                constant in wp-config.php.</p>
              <?php endif; ?>
            </td>
          </tr>
          <tr>
            <th scope="row"><label for="crabcrop-secret">Secret (signed mode)</label></th>
            <td>
              <input type="password" id="crabcrop-secret" class="regular-text"
                name="crabcrop_settings[secret]"
                value="<?php echo esc_attr($values['secret']); ?>"
                autocomplete="off"
                <?php disabled(crabcrop_wp_constant_for('secret') !== null); ?>>
              <?php if (crabcrop_wp_constant_for('secret')) : ?>
                <p class="description">Set by the
                <code><?php echo esc_html(crabcrop_wp_constant_for('secret')); ?></code>
                constant in wp-config.php.</p>
              <?php else : ?>
                <p class="description">The instance's <code>CRABCROP_SECRET</code>.
                Only used in signed mode; presets mode never reads it.</p>
              <?php endif; ?>
            </td>
          </tr>
          <tr>
            <th scope="row"><label for="crabcrop-size-map">Size mapping</label></th>
            <td>
              <textarea id="crabcrop-size-map" class="large-text code" rows="8"
                name="crabcrop_settings[size_map_text]"
                <?php disabled(crabcrop_wp_constant_for('size_map') !== null); ?>
                ><?php echo esc_textarea($values['size_map_text']); ?></textarea>
              <?php if (crabcrop_wp_constant_for('size_map')) : ?>
                <p class="description">Set by the
                <code><?php echo esc_html(crabcrop_wp_constant_for('size_map')); ?></code>
                constant in wp-config.php.</p>
              <?php else : ?>
                <p class="description">One mapping per line, as
                <code>size = target</code>. In presets mode a target is a
                preset name (<code>thumbnail = wp-thumb</code>). In signed
                mode a target is an operation string
                (<code>thumbnail = crop:150x150:smart</code>) or a preset
                reference (<code>medium = preset:card</code>). Sizes with
                no mapping keep their original URLs. Map
                <code>full</code> to also route the bare attachment URL.</p>
              <?php endif; ?>
            </td>
          </tr>
          <tr>
            <th scope="row">Intermediate sizes</th>
            <td>
              <label>
                <input type="checkbox" name="crabcrop_settings[disable_sizes]"
                  value="1" <?php checked($values['disable_sizes']); ?>
                  <?php disabled(crabcrop_wp_constant_for('disable_sizes') !== null); ?>>
                Stop generating intermediate sizes on upload (crabcrop
                renders every size on demand instead)
              </label>
              <?php if (crabcrop_wp_constant_for('disable_sizes')) : ?>
                <p class="description">Set by the
                <code><?php echo esc_html(crabcrop_wp_constant_for('disable_sizes')); ?></code>
                constant in wp-config.php.</p>
              <?php endif; ?>
            </td>
          </tr>
        </table>
        <?php submit_button(); ?>
      </form>
    </div>
    <?php
}

function crabcrop_wp_admin_init()
{
    register_setting('crabcrop_settings_group', 'crabcrop_settings', array(
        'type' => 'array',
        'sanitize_callback' => 'crabcrop_wp_sanitize_settings',
        'default' => array(),
    ));
}
add_action('admin_init', 'crabcrop_wp_admin_init');

function crabcrop_wp_admin_menu()
{
    add_options_page(
        'Crabcrop',
        'Crabcrop',
        'manage_options',
        'crabcrop',
        'crabcrop_wp_settings_page'
    );
}
add_action('admin_menu', 'crabcrop_wp_admin_menu');
