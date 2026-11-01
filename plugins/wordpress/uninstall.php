<?php
/**
 * Uninstall: remove the plugin's settings. Media and content are
 * never touched; the plugin only ever rewrote URLs at render time.
 */

if (!defined('WP_UNINSTALL_PLUGIN')) {
    exit;
}

delete_option('crabcrop_settings');
