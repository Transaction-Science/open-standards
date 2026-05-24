//! Browser-fingerprint collection and parsing.
//!
//! 3DS 2.x calls this "3DS Method invocation". Before the AReq is
//! built the cardholder's browser loads a small JavaScript snippet
//! served by the ACS at the URL returned in the DS version-check
//! `threeDSMethodURL`. The snippet collects device-side signals into
//! a hidden form and POSTs them back to the ACS, which uses them to
//! enrich its risk model when the AReq arrives shortly after.
//!
//! ## What we collect (EMVCo § 6.2.3)
//!
//! - `screen.width` / `screen.height` / `screen.colorDepth`
//! - `Date.getTimezoneOffset()`
//! - `navigator.userAgent`
//! - `navigator.languages` (joined with `,`)
//! - `navigator.javaEnabled()` (legacy; informational)
//! - Whether JavaScript is enabled (always true in the snippet path)
//!
//! ## What we deliberately do NOT collect
//!
//! - Plugin enumeration (deprecated in modern browsers)
//! - Hardware concurrency (entropy is too low to matter)
//! - WebGL/Canvas fingerprints (privacy-hostile and largely defeated
//!   by modern browser anti-fingerprinting)

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::risk::BrowserInfo;

/// Fingerprint-collection method discriminator. Browser-flow uses the
/// JS snippet; app-flow collects the equivalent envelope from the
/// 3DS SDK.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FingerprintMethod {
    /// `<script>` injected into the checkout page.
    BrowserScript,
    /// Native 3DS SDK on iOS / Android.
    AppSdk,
}

/// Parsed cardholder-side fingerprint envelope.
///
/// The shape mirrors [`BrowserInfo`] so the codec can lift this
/// straight into the AReq.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrowserFingerprint {
    /// `navigator.userAgent`.
    pub user_agent: String,
    /// Accept header copied from the original page request.
    pub accept_header: String,
    /// `navigator.language` (primary).
    pub language: String,
    /// All `navigator.languages`, comma-joined.
    pub languages_joined: String,
    /// Screen geometry.
    pub screen_width: u32,
    /// Screen geometry.
    pub screen_height: u32,
    /// Bits per pixel; serialized as a string per spec.
    pub color_depth: String,
    /// Minutes offset from UTC.
    pub timezone_offset: i32,
    /// True when `navigator.javaEnabled()` returned true.
    pub java_enabled: bool,
    /// True (always, since the snippet path runs).
    pub javascript_enabled: bool,
}

impl BrowserFingerprint {
    /// Build a [`BrowserInfo`] AReq-side envelope from this fingerprint.
    /// IP must be supplied separately; the snippet cannot read it.
    #[must_use]
    pub fn into_browser_info(self, ip: Option<String>) -> BrowserInfo {
        BrowserInfo {
            user_agent: self.user_agent,
            language: self.language,
            accept_header: self.accept_header,
            screen_width: self.screen_width,
            screen_height: self.screen_height,
            color_depth: self.color_depth,
            timezone_offset: self.timezone_offset,
            java_enabled: self.java_enabled,
            javascript_enabled: self.javascript_enabled,
            ip,
        }
    }

    /// Parse a fingerprint from a form-encoded body the snippet posts.
    ///
    /// Expected keys: `ua`, `accept`, `lang`, `langs`, `sw`, `sh`,
    /// `cd`, `tz`, `java`. Missing required keys yield
    /// [`Error::InvalidFingerprint`].
    pub fn parse_form(body: &str) -> Result<Self> {
        let mut ua = None;
        let mut accept = None;
        let mut lang = None;
        let mut langs = None;
        let mut sw = None;
        let mut sh = None;
        let mut cd = None;
        let mut tz = None;
        let mut java = None;
        for pair in body.split('&') {
            let (k, v) = pair
                .split_once('=')
                .ok_or(Error::InvalidFingerprint("malformed form pair"))?;
            let v = url_decode(v);
            match k {
                "ua" => ua = Some(v),
                "accept" => accept = Some(v),
                "lang" => lang = Some(v),
                "langs" => langs = Some(v),
                "sw" => sw = Some(v),
                "sh" => sh = Some(v),
                "cd" => cd = Some(v),
                "tz" => tz = Some(v),
                "java" => java = Some(v),
                _ => {}
            }
        }
        Ok(Self {
            user_agent: ua.ok_or(Error::InvalidFingerprint("missing ua"))?,
            accept_header: accept.unwrap_or_default(),
            language: lang.clone().ok_or(Error::InvalidFingerprint("missing lang"))?,
            languages_joined: langs.unwrap_or_else(|| lang.unwrap_or_default()),
            screen_width: sw
                .ok_or(Error::InvalidFingerprint("missing sw"))?
                .parse()
                .map_err(|_| Error::InvalidFingerprint("non-numeric sw"))?,
            screen_height: sh
                .ok_or(Error::InvalidFingerprint("missing sh"))?
                .parse()
                .map_err(|_| Error::InvalidFingerprint("non-numeric sh"))?,
            color_depth: cd.ok_or(Error::InvalidFingerprint("missing cd"))?,
            timezone_offset: tz
                .ok_or(Error::InvalidFingerprint("missing tz"))?
                .parse()
                .map_err(|_| Error::InvalidFingerprint("non-numeric tz"))?,
            java_enabled: matches!(java.as_deref(), Some("1" | "true")),
            javascript_enabled: true,
        })
    }
}

fn url_decode(s: &str) -> String {
    // Minimal `%xx` + `+` decoder; we don't pull in url just for this.
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hi = (bytes[i + 1] as char).to_digit(16);
                let lo = (bytes[i + 2] as char).to_digit(16);
                if let (Some(h), Some(l)) = (hi, lo) {
                    #[allow(clippy::cast_possible_truncation)]
                    out.push(((h << 4) | l) as u8);
                    i += 3;
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// JavaScript snippet the 3DS Server emits into the merchant checkout
/// page. The snippet posts the fingerprint to `post_url` which must be
/// the operator's collector endpoint (the operator forwards the body
/// to the ACS's `threeDSMethodURL`).
#[must_use]
pub fn fingerprint_collector_script(post_url: &str, three_ds_server_trans_id: &str) -> String {
    // We render via format! rather than an embedded template because
    // we only have two interpolation points and want to keep the
    // output stable (and lintable). Snippet is intentionally
    // dependency-free vanilla JS so it works under every CSP.
    format!(
        "(function(){{var d=document,b=d.body,f=d.createElement('form');\
f.method='POST';f.action={post_url:?};f.style.display='none';\
var add=function(k,v){{var i=d.createElement('input');i.type='hidden';\
i.name=k;i.value=String(v);f.appendChild(i)}};\
add('threeDSServerTransID',{trans:?});\
add('ua',navigator.userAgent);\
add('accept',d.contentType||'text/html');\
add('lang',navigator.language||'');\
add('langs',(navigator.languages||[]).join(','));\
add('sw',screen.width);add('sh',screen.height);\
add('cd',screen.colorDepth);\
add('tz',new Date().getTimezoneOffset());\
add('java',navigator.javaEnabled&&navigator.javaEnabled()?'1':'0');\
b.appendChild(f);f.submit();}})();",
        post_url = post_url,
        trans = three_ds_server_trans_id,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snippet_contains_required_keys() {
        let s = fingerprint_collector_script(
            "https://collector.example/3ds-method",
            "tid-abc",
        );
        assert!(s.contains("'threeDSServerTransID'"));
        assert!(s.contains("'ua'"));
        assert!(s.contains("'sw'"));
        assert!(s.contains("'sh'"));
        assert!(s.contains("'tz'"));
        assert!(s.contains("tid-abc"));
        assert!(s.contains("collector.example/3ds-method"));
    }

    #[test]
    fn parse_known_form_payload() {
        let body = "ua=Mozilla%2F5.0&accept=text%2Fhtml&lang=en-US&langs=en-US%2Cen\
                    &sw=1920&sh=1080&cd=24&tz=-420&java=0";
        let fp = BrowserFingerprint::parse_form(body).unwrap();
        assert_eq!(fp.user_agent, "Mozilla/5.0");
        assert_eq!(fp.language, "en-US");
        assert_eq!(fp.languages_joined, "en-US,en");
        assert_eq!(fp.screen_width, 1920);
        assert_eq!(fp.screen_height, 1080);
        assert_eq!(fp.color_depth, "24");
        assert_eq!(fp.timezone_offset, -420);
        assert!(!fp.java_enabled);
        assert!(fp.javascript_enabled);
    }

    #[test]
    fn parse_rejects_missing_required_key() {
        let body = "ua=Mozilla&accept=text&lang=en&sw=100&sh=100&cd=24";
        assert!(matches!(
            BrowserFingerprint::parse_form(body),
            Err(Error::InvalidFingerprint(_))
        ));
    }

    #[test]
    fn parse_rejects_non_numeric_dimension() {
        let body = "ua=M&accept=t&lang=en&sw=abc&sh=100&cd=24&tz=0&java=0";
        assert!(matches!(
            BrowserFingerprint::parse_form(body),
            Err(Error::InvalidFingerprint(_))
        ));
    }

    #[test]
    fn into_browser_info_preserves_fields() {
        let fp = BrowserFingerprint {
            user_agent: "UA".into(),
            accept_header: "text/html".into(),
            language: "en-US".into(),
            languages_joined: "en-US,en".into(),
            screen_width: 1280,
            screen_height: 720,
            color_depth: "24".into(),
            timezone_offset: 0,
            java_enabled: false,
            javascript_enabled: true,
        };
        let bi = fp.into_browser_info(Some("198.51.100.1".into()));
        assert_eq!(bi.screen_width, 1280);
        assert_eq!(bi.screen_height, 720);
        assert_eq!(bi.ip.as_deref(), Some("198.51.100.1"));
    }
}
