<?php
/**
 * Minimal WordPress stubs for the plugin test suite.
 *
 * Only the functions the plugin calls are stubbed, with semantics
 * close enough to WordPress for the code under test: a filter
 * registry where apply_filters chains the registered callbacks in
 * registration order, an options store, and the uploads/site URLs
 * the rewriter checks against. The suite's environment lives in
 * $GLOBALS['wp_stub'] so each test group can reshape it.
 */

$GLOBALS['wp_stub'] = array(
    'filters' => array(),   // hook => list of [callback, accepted_args]
    'actions' => array(),   // hook => list of callbacks
    'options' => array(),   // name => value
    'uploads_baseurl' => 'https://site.example.com/wp-content/uploads',
    'site_url' => 'https://site.example.com',
    'settings' => array(),  // register_setting records
    'pages' => array(),     // add_options_page records
);

/** Hooks the plugin itself registers; resets keep these. */
$GLOBALS['wp_stub_plugin_hooks'] = array(
    'wp_get_attachment_url',
    'wp_get_attachment_image_src',
    'wp_calculate_image_srcset',
    'intermediate_image_sizes_advanced',
    'admin_init',
    'admin_menu',
);

function wp_stub_reset()
{
    $filters = array();
    foreach ($GLOBALS['wp_stub']['filters'] as $hook => $entries) {
        if (in_array($hook, $GLOBALS['wp_stub_plugin_hooks'], true)) {
            $filters[$hook] = $entries;
        }
    }
    $actions = array();
    foreach ($GLOBALS['wp_stub']['actions'] as $hook => $entries) {
        if (in_array($hook, $GLOBALS['wp_stub_plugin_hooks'], true)) {
            $actions[$hook] = $entries;
        }
    }
    $GLOBALS['wp_stub'] = array(
        'filters' => $filters,
        'actions' => $actions,
        'options' => array(),
        'uploads_baseurl' => 'https://site.example.com/wp-content/uploads',
        'site_url' => 'https://site.example.com',
        'settings' => array(),
        'pages' => array(),
    );
}

function add_filter($hook, $callback, $priority = 10, $accepted_args = 1)
{
    $GLOBALS['wp_stub']['filters'][$hook][] = array($callback, $accepted_args);
    return true;
}

function add_action($hook, $callback, $priority = 10, $accepted_args = 1)
{
    $GLOBALS['wp_stub']['actions'][$hook][] = $callback;
    return add_filter($hook, $callback, $priority, $accepted_args);
}

function has_filter($hook, $callback = false)
{
    if (empty($GLOBALS['wp_stub']['filters'][$hook])) {
        return false;
    }
    if ($callback === false) {
        return true;
    }
    foreach ($GLOBALS['wp_stub']['filters'][$hook] as $entry) {
        if ($entry[0] === $callback) {
            return true;
        }
    }
    return false;
}

function apply_filters($hook, $value = null, ...$args)
{
    if (empty($GLOBALS['wp_stub']['filters'][$hook])) {
        return $value;
    }
    foreach ($GLOBALS['wp_stub']['filters'][$hook] as $entry) {
        list($callback, $accepted) = $entry;
        $call_args = array_slice(array_merge(array($value), $args), 0, max(1, $accepted));
        $value = call_user_func_array($callback, $call_args);
    }
    return $value;
}

function do_action($hook, ...$args)
{
    if (empty($GLOBALS['wp_stub']['actions'][$hook])) {
        return;
    }
    foreach ($GLOBALS['wp_stub']['actions'][$hook] as $callback) {
        call_user_func_array($callback, $args);
    }
}

function get_option($name, $default = false)
{
    return array_key_exists($name, $GLOBALS['wp_stub']['options'])
        ? $GLOBALS['wp_stub']['options'][$name] : $default;
}

function update_option($name, $value)
{
    $GLOBALS['wp_stub']['options'][$name] = $value;
    return true;
}

function delete_option($name)
{
    unset($GLOBALS['wp_stub']['options'][$name]);
    return true;
}

function wp_upload_dir()
{
    return array(
        'baseurl' => $GLOBALS['wp_stub']['uploads_baseurl'],
        'basedir' => '/var/www/wp-content/uploads',
    );
}

function get_site_url()
{
    return $GLOBALS['wp_stub']['site_url'];
}

function plugin_dir_path($file)
{
    return rtrim(dirname($file), '/') . '/';
}

/** A close-enough esc_url_raw: trim, drop control bytes, keep scheme. */
function esc_url_raw($url, $protocols = null)
{
    $url = trim((string) $url);
    $url = preg_replace('/[\x00-\x20\x7F]/', '', $url);
    return $url;
}

function esc_attr($text)
{
    return htmlspecialchars((string) $text, ENT_QUOTES, 'UTF-8');
}

function esc_html($text)
{
    return htmlspecialchars((string) $text, ENT_QUOTES, 'UTF-8');
}

function esc_textarea($text)
{
    return htmlspecialchars((string) $text, ENT_QUOTES, 'UTF-8');
}

function checked($checked, $current = true, $display = true)
{
    $result = $checked == $current ? ' checked="checked"' : '';
    if ($display) {
        echo $result;
    }
    return $result;
}

function disabled($disabled, $current = true, $display = true)
{
    $result = $disabled == $current ? ' disabled="disabled"' : '';
    if ($display) {
        echo $result;
    }
    return $result;
}

function current_user_can($capability)
{
    return true;
}

function register_setting($group, $name, $args = array())
{
    $GLOBALS['wp_stub']['settings'][$name] = array($group, $args);
}

function add_options_page($page_title, $menu_title, $capability, $slug, $callback)
{
    $GLOBALS['wp_stub']['pages'][$slug] = $callback;
    return 'settings_page_' . $slug;
}

function settings_fields($group)
{
    return '';
}

function submit_button()
{
    return '';
}
