//! ISO/IEC 18013-5 + 18013-7 mobile driving licence (mDL) and broader
//! mDOC ingestion for Smart Byte.
//!
//! `smart-byte-mdl` packages the binary, CBOR-based credential format
//! deployed in Apple Wallet, Google Wallet, and Samsung Wallet for US
//! state driver's licences, and in the EU eIDAS 2.0 Architecture and
//! Reference Framework (EUDIW). It pulls both the issuance and
//! verification sides of that format into the Smart Byte substrate.
//!
//! # Surface
//!
//! * [`mdoc`] — the [`MobileDoc`] type and CBOR codec.
//! * [`namespace`] — canonical mDL data elements for `org.iso.18013.5.1`,
//!   the AAMVA US add-on namespace, and the EU eIDAS namespaces, with
//!   strongly typed accessors.
//! * [`issuer`] — [`Issuer`] mints a [`MobileDoc`] by building per-item
//!   IssuerSignedItems with random salts, hashing each into a Mobile
//!   Security Object (MSO), and signing the MSO with COSE_Sign1.
//! * [`verifier`] — [`Verifier`] checks the issuer COSE_Sign1 against a
//!   set of trust anchors, validates the MSO's validity window,
//!   re-hashes every revealed item against the MSO digest tree, and
//!   (if a `device_signed` half is present) verifies the device
//!   signature against the key bound in the MSO.
//! * [`selective_disclosure`] — the salted-hash projection used by
//!   ISO 18013-5: a holder reveals only the elements asked for, signs a
//!   device-authentication structure over the session transcript, and
//!   the verifier reconstructs the digest tree from the remaining MSO
//!   commitments.
//! * [`session_transcript`] — the [`SessionTranscript`] primitive
//!   shared by 18013-5 device-engagement and 18013-7 online flows.
//! * [`device_engagement`] — the QR/NFC engagement encoder and `mdoc:`
//!   URL form.
//! * [`online_presentment`] — the 18013-7 reader-side `ItemsRequest`
//!   and `DeviceResponse` wire types.
//! * [`cargo_bridge`] — packs an mdoc into a Smart Byte envelope as a
//!   `Cargo::Custom { type_uri = "urn:smart-byte:cargo:mdoc:v1", body =
//!   cbor(mdoc) }` payload so the envelope's SAID and signature
//!   machinery work unchanged.
//!
//! # Scope notes
//!
//! * ECDSA on NIST P-256 (COSE alg `-7`, ES256) is the only fully
//!   implemented signature path. ES256 is the ISO 18013-5 baseline and
//!   the only curve currently mandated by AAMVA. ES384 and ES512 wire
//!   shapes are recognised; verification slots for them are
//!   intentionally narrow so a downstream crate can fill them in.
//! * X.509 chain validation is delegated to a downstream verifier;
//!   this crate exposes raw trust anchors (subject public keys keyed by
//!   COSE algorithm). The certificate chain still travels intact in
//!   the COSE_Sign1 unprotected header (`x5chain`, label 33).
//! * Transport (BLE, NFC, Wi-Fi Aware, HTTPS) is platform-specific and
//!   intentionally not shipped. [`device_engagement`] and
//!   [`session_transcript`] expose the primitives an agent crate needs
//!   to wire transport in.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod cargo_bridge;
pub mod device_engagement;
pub mod error;
pub mod issuer;
pub mod mdoc;
pub mod namespace;
pub mod online_presentment;
pub mod selective_disclosure;
pub mod session_transcript;
pub mod verifier;

pub use cargo_bridge::{
    MDL_CARGO_TYPE_URI, mdoc_envelope, mdoc_from_envelope, mdoc_said,
};
pub use device_engagement::{DeviceEngagement, DeviceRetrievalMethod};
pub use error::MdlError;
pub use issuer::{
    COSE_ALG_ES256, DIGEST_ALG_SHA256, Issuer, IssuerKey,
    MobileSecurityObject, MSO_VERSION, ValidityInfo,
};
pub use mdoc::{
    CoseMac0, CoseSign1, DeviceAuth, DeviceSigned, IssuerSigned,
    IssuerSignedItem, MobileDoc, TAG_ENCODED_CBOR, decode_cbor, encode_cbor,
};
pub use namespace::{
    MdlClaims, NS_EIDAS_MDL, NS_EIDAS_PID, NS_MDL, NS_MDL_AAMVA, aamva,
    eidas_pid, mdl,
};
pub use online_presentment::{DeviceResponse, ItemsRequest};
pub use selective_disclosure::{DeviceSigner, P256DeviceSigner, present};
pub use session_transcript::SessionTranscript;
pub use verifier::{
    FixedTime, SystemTime, TimeProvider, TrustAnchor, VerifiedClaims, Verifier,
};
