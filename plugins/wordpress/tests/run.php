<?php
/**
 * Test suite for the crabcrop WordPress plugin.
 *
 * Run with: php tests/run.php
 *
 * With no argument, every group runs in its own PHP process (plugin
 * state and wp-config constants are per-process) and the exit code
 * aggregates. A single group can be run directly:
 * php tests/run.php rewrite
 *
 * The suite stubs the WordPress functions the plugin calls (see
 * wp-stubs.php); the URL builder is additionally verified against
 * the canonical spec vectors in spec/test-vectors.json, the same
 * vectors every other implementation runs.
 */

$GROUPS = array(
    'vectors',
    'builder',
    'config_options',
    'config_filters',
    'config_constants',
    'rewrite',
    'settings',
);

if (!isset($argv[1])) {
    $failed = 0;
    foreach ($GROUPS as $group) {
        $cmd = escapeshellarg(PHP_BINARY) . ' ' . escapeshellarg(__FILE__)
            . ' ' . escapeshellarg($group);
        passthru($cmd, $code);
        if ($code !== 0) {
            $failed = 1;
        }
    }
    exit($failed);
}

$GROUP = $argv[1];
if (!in_array($GROUP, $GROUPS, true)) {
    fwrite(STDERR, "unknown group: $GROUP\n");
    exit(2);
}

define('ABSPATH', '/tmp/wp-stub/');

// The constants group defines its wp-config constants before the
// plugin reads any configuration.
if ($GROUP === 'config_constants') {
    define('CRABCROP_WP_BASE_URL', 'https://const.example.com');
    define('CRABCROP_WP_MODE', 'signed');
    define('CRABCROP_WP_SECRET', 'const-secret');
    define('CRABCROP_WP_DISABLE_SIZES', true);
}

require dirname(__FILE__) . '/wp-stubs.php';
require dirname(dirname(__FILE__)) . '/crabcrop.php';

$TESTS = 0;
$FAILURES = 0;

function check($condition, $name)
{
    global $TESTS, $FAILURES;
    $TESTS++;
    if ($condition) {
        echo "ok      $name\n";
    } else {
        $FAILURES++;
        echo "FAIL    $name\n";
    }
}

function eq($expected, $actual, $name)
{
    global $TESTS, $FAILURES;
    $TESTS++;
    if ($expected === $actual) {
        echo "ok      $name\n";
    } else {
        $FAILURES++;
        echo "FAIL    $name\n";
        echo "        expected: " . var_export($expected, true) . "\n";
        echo "        actual:   " . var_export($actual, true) . "\n";
    }
}

function throws_invalid($fn, $name)
{
    try {
        $fn();
    } catch (InvalidArgumentException $e) {
        check(true, $name);
        return;
    }
    check(false, $name);
}

function set_options($options)
{
    wp_stub_reset();
    update_option('crabcrop_settings', $options);
    crabcrop_wp_reset_config();
}

$UPLOADS = 'https://site.example.com/wp-content/uploads';
$PHOTO = $UPLOADS . '/2026/11/photo.jpg';

/* ------------------------------------------------------------------ */

if ($GROUP === 'vectors') {
    $file = dirname(__FILE__, 4) . '/spec/test-vectors.json';
    $vectors = json_decode(file_get_contents($file), true);
    check(is_array($vectors) && count($vectors) > 0, 'spec vectors load');
    $signed = 0;
    $refused = 0;
    foreach ($vectors as $vector) {
        if (!empty($vector['sealed'])) {
            continue; // Sealing is out of scope for a URL builder.
        }
        if (isset($vector['refuses'])) {
            // Builder refusals (issue #472): signed and unsafe
            // builds both refuse, naming the input, and a
            // character a browser rewrites comes with its
            // percent-encoding.
            $character = $vector['refuses']['character'];
            foreach (array('s3cr3t', null) as $secret) {
                $message = null;
                try {
                    CrabcropUrl::build(
                        'https://images.example.com',
                        $secret,
                        $vector['operations'],
                        $vector['source']
                    );
                } catch (InvalidArgumentException $e) {
                    $message = $e->getMessage();
                }
                $ok = $message !== null
                    && strpos($message, $vector['refuses']['input'] . ' contains') === 0;
                if ($ok && strpos('"<>\\^`{|}', $character) !== false) {
                    $hint = sprintf('percent-encode it as %%%02X)', ord($character));
                    $ok = substr($message, -strlen($hint)) === $hint;
                }
                check($ok, 'vector: ' . $vector['description'] . ($secret === null ? ' (unsafe)' : ''));
            }
            $refused++;
            continue;
        }
        $signed++;
        eq(
            $vector['signature'],
            CrabcropUrl::sign($vector['secret'], $vector['remainder']),
            'vector: ' . $vector['description']
        );
    }
    check($signed >= 20, "signature vectors covered ($signed)");
    check($refused >= 17, "refusal vectors covered ($refused)");

    eq(
        'https://images.example.com/3c1f6c180116661877bd4c85fd201a178087d74c1cc188cc584e819d901308c6'
            . '/300x200/https://example.com/photo.jpg',
        CrabcropUrl::build(
            'https://images.example.com',
            'your-secret',
            '300x200',
            'https://example.com/photo.jpg'
        ),
        'build: README worked example'
    );
    eq(
        'https://images.example.com/4ccaf0251e8899792852954b375a4a0e004a6e3f79de81eb598fe4ec818fabad'
            . '/preset:og-card/filters:grayscale/https://example.com/photo.jpg',
        CrabcropUrl::build(
            'https://images.example.com',
            's3cr3t',
            'preset:og-card/filters:grayscale',
            'https://example.com/photo.jpg'
        ),
        'build: preset reference and filter chain vector'
    );
    eq(
        'https://images.example.com/878f5cb69f5ac5770161176099fcdd64dcce17d6e6f846a3108ffacfa4644386'
            . '/exp:1893456000/300x200/https://example.com/photo.jpg',
        CrabcropUrl::build(
            'https://images.example.com',
            's3cr3t',
            '300x200',
            'https://example.com/photo.jpg',
            1893456000
        ),
        'build: expiry is emitted first (vector)'
    );
}

if ($GROUP === 'builder') {
    eq(
        'https://images.example.com/unsafe/preset:card/https://example.com/photo.jpg',
        CrabcropUrl::build('https://images.example.com/', null, 'preset:card', 'https://example.com/photo.jpg'),
        'null secret builds the unsafe form; base trailing slash trimmed'
    );
    eq(
        'https://images.example.com/unsafe/https://example.com/photo.jpg',
        CrabcropUrl::build('https://images.example.com', null, '', 'https://example.com/photo.jpg'),
        'empty operations: remainder is the bare source'
    );
    $empty_secret = CrabcropUrl::build('https://images.example.com', '', '300x200', 'https://example.com/photo.jpg');
    eq(
        'https://images.example.com/' . hash_hmac('sha256', '300x200/https://example.com/photo.jpg', '')
            . '/300x200/https://example.com/photo.jpg',
        $empty_secret,
        'empty-string secret is a real key and signs (only null is unsafe)'
    );
    $keyed = CrabcropUrl::build('https://images.example.com', 's3cr3t', '300x200', 'https://example.com/photo.jpg', null, 'k2');
    eq(
        'https://images.example.com/k2:' . hash_hmac('sha256', '300x200/https://example.com/photo.jpg', 's3cr3t')
            . '/300x200/https://example.com/photo.jpg',
        $keyed,
        'key id stamps a <key-id>: prefix on the signature segment'
    );

    throws_invalid(function () {
        CrabcropUrl::build('https://images.example.com', 's', '300x200', 'https://example.com/a b.jpg');
    }, 'rejects a source with whitespace');
    throws_invalid(function () {
        CrabcropUrl::build('https://images.example.com', 's', '300x200', 'https://example.com/a.jpg?x=1');
    }, 'rejects a source with a query string');
    throws_invalid(function () {
        CrabcropUrl::build('https://images.example.com', 's', "300x200\n", 'https://example.com/a.jpg');
    }, 'rejects operations with a control character');
    throws_invalid(function () {
        CrabcropUrl::build('https://images.example.com', 's', '300x200', "https://example.com/\xc3\xbc.jpg");
    }, 'rejects a non-ASCII source');
    throws_invalid(function () {
        CrabcropUrl::build('https://images.example.com', 's', '300x200', '');
    }, 'rejects an empty source');
    throws_invalid(function () {
        CrabcropUrl::build('https://images.example.com', 's', '300x200', 'https://example.com/a.jpg', -1);
    }, 'rejects a negative expiry');
    throws_invalid(function () {
        CrabcropUrl::build('https://images.example.com', 's', '300x200', 'https://example.com/a.jpg', 1.5);
    }, 'rejects a non-integer expiry');
    throws_invalid(function () {
        CrabcropUrl::build('https://images.example.com', 's', '300x200', 'https://example.com/a.jpg', null, 'bad key!');
    }, 'rejects a malformed key id');
    throws_invalid(function () {
        CrabcropUrl::build('https://images.example.com', 's', '300x200', 'https://example.com/a.jpg', null, str_repeat('k', 33));
    }, 'rejects a key id past 32 characters');
}

if ($GROUP === 'config_options') {
    set_options(array(
        'base_url' => 'https://images.example.com/',
        'mode' => 'signed',
        'secret' => 'option-secret',
        'size_map' => array('thumbnail' => 'crop:150x150'),
        'disable_sizes' => true,
    ));
    $config = crabcrop_wp_config();
    eq('https://images.example.com', $config['base_url'], 'options: base URL read, trailing slash trimmed');
    eq('signed', $config['mode'], 'options: mode');
    eq('option-secret', $config['secret'], 'options: secret');
    eq(array('thumbnail' => 'crop:150x150'), $config['size_map'], 'options: size map');
    eq(true, $config['disable_sizes'], 'options: disable sizes');

    set_options(array());
    $config = crabcrop_wp_config();
    eq('', $config['base_url'], 'defaults: no base URL (plugin inert)');
    eq('presets', $config['mode'], 'defaults: presets mode');
    eq(array(), $config['size_map'], 'defaults: empty size map');

    set_options(array('mode' => 'bogus'));
    eq('presets', crabcrop_wp_config()['mode'], 'an unknown mode falls back to presets');
}

if ($GROUP === 'config_filters') {
    set_options(array(
        'base_url' => 'https://options.example.com',
        'mode' => 'presets',
        'secret' => 'option-secret',
        'size_map' => array('thumbnail' => 'from-options'),
        'disable_sizes' => false,
    ));
    add_filter('crabcrop_wp_base_url', function ($v) {
        return 'https://filter.example.com';
    });
    add_filter('crabcrop_wp_size_map', function ($v) {
        return array('medium' => 'from-filter');
    });
    add_filter('crabcrop_wp_disable_sizes', function ($v) {
        return true;
    });
    crabcrop_wp_reset_config();
    $config = crabcrop_wp_config();
    eq('https://filter.example.com', $config['base_url'], 'filter beats the settings screen (base URL)');
    eq('presets', $config['mode'], 'unfiltered setting still comes from the screen');
    eq(array('medium' => 'from-filter'), $config['size_map'], 'filter beats the settings screen (size map)');
    eq(true, $config['disable_sizes'], 'filter beats the settings screen (disable sizes)');

    // A filter whose chained result is not an array is ignored as a
    // whole: the screen value survives.
    wp_stub_reset();
    update_option('crabcrop_settings', array(
        'size_map' => array('thumbnail' => 'from-options'),
    ));
    add_filter('crabcrop_wp_size_map', function ($v) {
        return 'not-an-array';
    });
    crabcrop_wp_reset_config();
    eq(
        array('thumbnail' => 'from-options'),
        crabcrop_wp_config()['size_map'],
        'a filter returning a non-array size map is ignored'
    );
}

if ($GROUP === 'config_constants') {
    set_options(array(
        'base_url' => 'https://options.example.com',
        'mode' => 'presets',
        'secret' => 'option-secret',
        'size_map' => array('thumbnail' => 'from-options'),
        'disable_sizes' => false,
    ));
    add_filter('crabcrop_wp_base_url', function ($v) {
        return 'https://filter.example.com';
    });
    add_filter('crabcrop_wp_mode', function ($v) {
        return 'presets';
    });
    crabcrop_wp_reset_config();
    $config = crabcrop_wp_config();
    eq('https://const.example.com', $config['base_url'], 'constant beats filter and screen (base URL)');
    eq('signed', $config['mode'], 'constant beats filter and screen (mode)');
    eq('const-secret', $config['secret'], 'constant supplies the secret');
    eq(true, $config['disable_sizes'], 'constant supplies disable sizes');
    eq('CRABCROP_WP_BASE_URL', crabcrop_wp_constant_for('base_url'), 'constant_for names the base URL constant');
    eq(null, crabcrop_wp_constant_for('key_id'), 'constant_for: key id has no settings field');
}

if ($GROUP === 'rewrite') {
    $base = 'https://images.example.com';

    // Presets mode.
    set_options(array(
        'base_url' => $base,
        'mode' => 'presets',
        'size_map' => array('thumbnail' => 'wp-thumb', 'medium' => 'wp-medium'),
    ));
    eq(
        $PHOTO,
        apply_filters('wp_get_attachment_url', $PHOTO, 42),
        'presets: attachment URL untouched while full is unmapped'
    );
    $img = apply_filters(
        'wp_get_attachment_image_src',
        array($UPLOADS . '/2026/11/photo-300x200.jpg', 300, 200, true),
        42,
        'medium',
        false
    );
    eq(
        $base . '/unsafe/preset:wp-medium/' . $UPLOADS . '/2026/11/photo-300x200.jpg',
        $img[0],
        'presets: mapped size rewrites to the unsafe preset URL'
    );
    $img = apply_filters(
        'wp_get_attachment_image_src',
        array($UPLOADS . '/2026/11/photo-1024x768.jpg', 1024, 768, true),
        42,
        'large',
        false
    );
    eq($UPLOADS . '/2026/11/photo-1024x768.jpg', $img[0], 'presets: unmapped size keeps its original URL');
    $img = apply_filters(
        'wp_get_attachment_image_src',
        array('https://other.example.com/photo-300x200.jpg', 300, 200, true),
        42,
        'medium',
        false
    );
    eq('https://other.example.com/photo-300x200.jpg', $img[0], 'presets: a foreign host is never rewritten');

    $meta = array(
        'file' => '2026/11/photo.jpg',
        'sizes' => array(
            'thumbnail' => array('file' => 'photo-150x150.jpg', 'width' => 150, 'height' => 150),
            'medium' => array('file' => 'photo-300x200.jpg', 'width' => 300, 'height' => 200),
            'large' => array('file' => 'photo-1024x683.jpg', 'width' => 1024, 'height' => 683),
        ),
    );
    $sources = array(
        150 => array('url' => $UPLOADS . '/2026/11/photo-150x150.jpg', 'descriptor' => 'w', 'value' => 150),
        300 => array('url' => $UPLOADS . '/2026/11/photo-300x200.jpg', 'descriptor' => 'w', 'value' => 300),
        1024 => array('url' => $UPLOADS . '/2026/11/photo-1024x683.jpg', 'descriptor' => 'w', 'value' => 1024),
    );
    $out = apply_filters('wp_calculate_image_srcset', $sources, array(300, 200), $sources[300]['url'], $meta, 42);
    eq($base . '/unsafe/preset:wp-thumb/' . $PHOTO, $out[150]['url'], 'presets srcset: thumbnail via its preset, from the original');
    eq($base . '/unsafe/preset:wp-medium/' . $PHOTO, $out[300]['url'], 'presets srcset: medium via its preset, from the original');
    eq($sources[1024]['url'], $out[1024]['url'], 'presets srcset: unmapped large keeps its original URL');

    // Presets mode with full mapped.
    set_options(array(
        'base_url' => $base,
        'mode' => 'presets',
        'size_map' => array('full' => 'wp-full'),
    ));
    eq(
        $base . '/unsafe/preset:wp-full/' . $PHOTO,
        apply_filters('wp_get_attachment_url', $PHOTO, 42),
        'presets: mapping full routes the bare attachment URL'
    );

    // Signed mode.
    set_options(array(
        'base_url' => $base,
        'mode' => 'signed',
        'secret' => 'test-secret',
        'size_map' => array('thumbnail' => 'crop:150x150:smart'),
    ));
    $img = apply_filters(
        'wp_get_attachment_image_src',
        array($UPLOADS . '/2026/11/photo-150x150.jpg', 150, 150, true),
        42,
        'thumbnail',
        false
    );
    $remainder = 'crop:150x150:smart/' . $UPLOADS . '/2026/11/photo-150x150.jpg';
    eq(
        $base . '/' . hash_hmac('sha256', $remainder, 'test-secret') . '/' . $remainder,
        $img[0],
        'signed: mapped size signs its operation string'
    );
    $out = apply_filters('wp_calculate_image_srcset', $sources, array(300, 200), $sources[300]['url'], $meta, 42);
    $remainder = 'crop:150x150:smart/' . $PHOTO;
    eq(
        $base . '/' . hash_hmac('sha256', $remainder, 'test-secret') . '/' . $remainder,
        $out[150]['url'],
        'signed srcset: mapped size signs from the original'
    );
    eq($sources[300]['url'], $out[300]['url'], 'signed srcset: a registered but unmapped size keeps its URL');
    $wide = array(
        1200 => array('url' => $PHOTO, 'descriptor' => 'w', 'value' => 1200),
    );
    $out = apply_filters('wp_calculate_image_srcset', $wide, array(1200, 800), $PHOTO, $meta, 42);
    $remainder = '1200x0/' . $PHOTO;
    eq(
        $base . '/' . hash_hmac('sha256', $remainder, 'test-secret') . '/' . $remainder,
        $out[1200]['url'],
        'signed srcset: a width no size owns becomes a width-only resize'
    );

    // Signed mode without a secret is inert.
    set_options(array('base_url' => $base, 'mode' => 'signed', 'secret' => '', 'size_map' => array('thumbnail' => 'crop:150x150:smart')));
    $img = apply_filters(
        'wp_get_attachment_image_src',
        array($UPLOADS . '/2026/11/photo-150x150.jpg', 150, 150, true),
        42,
        'thumbnail',
        false
    );
    eq($UPLOADS . '/2026/11/photo-150x150.jpg', $img[0], 'signed mode without a secret leaves URLs alone');

    // No base URL: fully inert.
    set_options(array('mode' => 'presets', 'size_map' => array('medium' => 'wp-medium')));
    $img = apply_filters(
        'wp_get_attachment_image_src',
        array($UPLOADS . '/2026/11/photo-300x200.jpg', 300, 200, true),
        42,
        'medium',
        false
    );
    eq($UPLOADS . '/2026/11/photo-300x200.jpg', $img[0], 'no base URL: plugin is inert');

    // The sizes toggle.
    set_options(array('base_url' => $base, 'disable_sizes' => true));
    eq(
        array(),
        apply_filters('intermediate_image_sizes_advanced', array('thumbnail' => array('width' => 150))),
        'disable sizes on: no intermediate sizes are generated'
    );
    set_options(array('base_url' => $base, 'disable_sizes' => false));
    eq(
        array('thumbnail' => array('width' => 150)),
        apply_filters('intermediate_image_sizes_advanced', array('thumbnail' => array('width' => 150))),
        'disable sizes off: sizes pass through'
    );

    // The final URL filter sees every rewrite.
    set_options(array(
        'base_url' => $base,
        'mode' => 'presets',
        'size_map' => array('medium' => 'wp-medium'),
    ));
    add_filter('crabcrop_wp_url', function ($url, $size, $id) {
        return $url . '#via-filter';
    }, 10, 3);
    $img = apply_filters(
        'wp_get_attachment_image_src',
        array($UPLOADS . '/2026/11/photo-300x200.jpg', 300, 200, true),
        42,
        'medium',
        false
    );
    eq(
        $base . '/unsafe/preset:wp-medium/' . $UPLOADS . '/2026/11/photo-300x200.jpg#via-filter',
        $img[0],
        'the crabcrop_wp_url filter post-processes built URLs'
    );
}

if ($GROUP === 'settings') {
    $clean = crabcrop_wp_sanitize_settings(array(
        'base_url' => 'https://images.example.com/',
        'mode' => 'signed',
        'secret' => 'shh',
        'size_map_text' => "# comment line\n\nthumbnail = crop:150x150:smart\nmedium = preset:card\nbogus line\nBad Size = 300x200\nlarge = has space\n",
        'disable_sizes' => '1',
    ));
    eq('https://images.example.com', $clean['base_url'], 'sanitize: base URL cleaned and unslashed');
    eq('signed', $clean['mode'], 'sanitize: mode kept');
    eq('shh', $clean['secret'], 'sanitize: secret kept');
    eq(
        array('thumbnail' => 'crop:150x150:smart', 'medium' => 'preset:card'),
        $clean['size_map'],
        'sanitize: map parses, comments and bad lines dropped'
    );
    eq(true, $clean['disable_sizes'], 'sanitize: disable sizes checkbox');

    $clean = crabcrop_wp_sanitize_settings(array(
        'base_url' => 'javascript:alert(1)',
        'mode' => 'bogus',
        'size_map_text' => "thumbnail = wp-thumb\nmedium = not a preset\n",
    ));
    eq('', $clean['base_url'], 'sanitize: a non-http base URL is dropped');
    eq('presets', $clean['mode'], 'sanitize: unknown mode falls back to presets');
    eq(
        array('thumbnail' => 'wp-thumb'),
        $clean['size_map'],
        'sanitize: presets mode accepts preset names only'
    );

    eq(
        "thumbnail = wp-thumb\nmedium = wp-medium",
        crabcrop_wp_size_map_to_text(array('thumbnail' => 'wp-thumb', 'medium' => 'wp-medium')),
        'size map serializes back to settings text'
    );
    eq(
        array('a' => 'b'),
        crabcrop_wp_parse_size_map('a=b', 'presets'),
        'parse: tight spacing works'
    );

    // Admin wiring: the actions register the setting and the page.
    do_action('admin_init');
    check(isset($GLOBALS['wp_stub']['settings']['crabcrop_settings']), 'admin_init registers the setting');
    eq(
        'crabcrop_wp_sanitize_settings',
        $GLOBALS['wp_stub']['settings']['crabcrop_settings'][1]['sanitize_callback'],
        'the registered sanitize callback is the tested one'
    );
    do_action('admin_menu');
    check(isset($GLOBALS['wp_stub']['pages']['crabcrop']), 'admin_menu adds the settings page');
}

echo "---\n$GROUP: $TESTS checks, $FAILURES failures\n";
exit($FAILURES === 0 ? 0 : 1);
