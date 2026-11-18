//! The cryptography under every HTTPS connection the server
//! opens, to a source, a bucket or a timestamp authority
//! (issue #742): one rustls crypto provider, ring, and the
//! build compiles no other.
//!
//! ring is the provider the C2PA SDK's HTTP stack brings:
//! c2pa builds ureq with its `rustls` feature, which turns
//! ring on, and no feature of crabcrop's turns it off.
//! reqwest's own `rustls` feature would add aws-lc-rs, and
//! the AWS-LC C library under it, so reqwest is built with
//! `rustls-no-provider` and takes the process default. It
//! panics when it builds a client and no default is
//! installed, so every reqwest client starts from
//! [`client_builder`] (clippy.toml holds the rest of the
//! code to that). ureq takes the same default, and uses
//! ring when none is installed.

use std::sync::Once;

/// Install ring as the process default crypto provider. The
/// first call installs it, and a provider installed before
/// that, by a program embedding the library, stays.
pub fn install_crypto_provider() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        // Refused only when a provider is already installed.
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

/// `reqwest::Client::builder()`, after
/// [`install_crypto_provider`]. Every reqwest client the
/// server builds starts here.
#[expect(
    clippy::disallowed_methods,
    reason = "the one place a reqwest client starts"
)]
pub fn client_builder() -> reqwest::ClientBuilder {
    install_crypto_provider();
    reqwest::Client::builder()
}

#[cfg(test)]
mod tests {
    use rustls::crypto::CryptoProvider;

    /// The cipher suites and key exchange groups a provider
    /// offers, in its order of preference.
    fn offer(provider: &CryptoProvider) -> (Vec<rustls::CipherSuite>, Vec<rustls::NamedGroup>) {
        (
            provider.cipher_suites.iter().map(|s| s.suite()).collect(),
            provider.kx_groups.iter().map(|g| g.name()).collect(),
        )
    }

    #[test]
    fn a_client_takes_ring_as_the_process_default() {
        // reqwest brings no provider: a client builds only
        // once a process default is installed.
        let client = super::client_builder().build();
        assert!(client.is_ok(), "{:?}", client.err());
        let installed =
            CryptoProvider::get_default().expect("a process default provider is installed");
        assert_eq!(
            offer(installed),
            offer(&rustls::crypto::ring::default_provider())
        );
    }
}
