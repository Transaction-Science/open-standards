//! OpenID for Verifiable Credentials (OID4VC) — issuance + presentation
//! protocols layered on top of [`smart_byte_vc`].
//!
//! This crate ingests the deployed OpenID Foundation profile family
//! that adapts OAuth 2.0 + OpenID Connect to verifiable-credential
//! workflows:
//!
//! * **OID4VCI** (OpenID for Verifiable Credential Issuance) draft 13.
//!   Issuer + Authorisation-Server metadata, credential offers with
//!   pre-authorized-code and authorization-code grants, the token /
//!   nonce / credential / notification endpoints, batch issuance, and
//!   DPoP-bound access tokens (RFC 9449).
//! * **OID4VP** (OpenID for Verifiable Presentations) draft 23. The
//!   verifier-side authorization request, DCQL queries (new in draft
//!   23), and legacy DIF Presentation Exchange 2.0.
//! * **SIOPv2** (Self-Issued OpenID Provider v2). Wallet-as-OP
//!   `id_token` issuance.
//! * **Status mechanisms**: Bitstring Status List 2021 (re-exported
//!   from [`smart_byte_vc`]) and IETF Token Status List.
//! * **Profiles**: HAIP (High-Assurance Interoperability Profile) and
//!   the EUDI Wallet ARF profile.
//!
//! Composition with the rest of the substrate is intentionally loose:
//! credentials returned over the wire are SD-JWT VCs, mdoc CBOR, or
//! W3C VCs — they round-trip through `smart-byte-vc`,
//! `smart-byte-mdl`, and `smart-byte-bbs` respectively. This crate
//! supplies *only* the protocol skin.
//!
//! ## Module map
//!
//! | Module | Responsibility |
//! | --- | --- |
//! | [`issuer`] | Credential Issuer + AS metadata. |
//! | [`offer`] | Credential offers (pre-auth + auth-code). |
//! | [`wallet`] | Wallet-side state machine. |
//! | [`token`] | Token endpoint + DPoP claims. |
//! | [`credential`] | Credential endpoint + nonce endpoint. |
//! | [`notification`] | Notification endpoint. |
//! | [`verifier`] | OID4VP authorization request + response. |
//! | [`dcql`] | Digital Credentials Query Language. |
//! | [`presentation`] | Presentation Definition + Submission (PE 2.0). |
//! | [`siop`] | Self-Issued OpenID Provider v2. |
//! | [`status_list`] | Token Status List + W3C Bitstring re-export. |
//! | [`haip`] | High-Assurance Interoperability Profile checks. |
//! | [`eudi`] | EUDI Wallet ARF profile checks. |

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod credential;
pub mod dcql;
pub mod error;
pub mod eudi;
pub mod haip;
pub mod issuer;
pub mod notification;
pub mod offer;
pub mod presentation;
pub mod siop;
pub mod status_list;
pub mod token;
pub mod verifier;
pub mod wallet;

pub use credential::{
    CredentialProof, CredentialProofs, CredentialRequest, CredentialResponse,
    NonceResponse, credential_request_jwt,
};
pub use dcql::{
    ClaimQuery, CredentialMeta, CredentialQuery, CredentialSet, DcqlQuery,
    PathSegment, evaluate_claim, evaluate_credential,
};
pub use error::OidcError;
pub use eudi::{
    EUDI_MDL_DOCTYPE, EUDI_MDL_VCT, EUDI_PID_DOCTYPE, EUDI_PID_VCT, EudiCheck,
    EudiReport,
};
pub use haip::{HAIP_ALG_ES256, HAIP_FORMAT_MSO_MDOC, HAIP_FORMAT_SD_JWT_VC, HaipCheck, HaipReport};
pub use issuer::{
    AuthorizationServerMetadata, CredentialConfiguration, CredentialDefinition,
    CredentialIssuer, DisplayInfo, IssuerMetadata, ProofTypeMetadata,
};
pub use notification::{
    NotificationErrorResponse, NotificationEvent, NotificationRequest,
};
pub use offer::{
    AuthorizationCodeGrant, CredentialOffer, OfferGrants, PreAuthorizedGrant,
    TxCodeSpec,
};
pub use presentation::{
    Constraints, Field, InputDescriptor, PresentationDefinition,
    PresentationSubmission, SubmissionDescriptor, simple_definition,
};
pub use siop::{
    SIOP_V2_ISS, SUBJECT_SYNTAX_DID_JWK, SUBJECT_SYNTAX_DID_KEY,
    SUBJECT_SYNTAX_JWK_THUMBPRINT, SiopAuthRequest, SiopIdToken,
};
pub use status_list::{
    BitstringStatusList, STATUS_APPLICATION_SPECIFIC, STATUS_INVALID,
    STATUS_SUSPENDED, STATUS_VALID, StatusPurpose, TokenStatusList,
    TokenStatusListBytes, TokenStatusReference, check_bitstring_status,
};
pub use token::{
    DpopClaims, TokenErrorCode, TokenErrorResponse, TokenRequest, TokenResponse,
    validate_dpop_claims,
};
pub use verifier::{
    CLIENT_ID_SCHEME_DID, CLIENT_ID_SCHEME_REDIRECT_URI,
    CLIENT_ID_SCHEME_VERIFIER_ATTESTATION, CLIENT_ID_SCHEME_X509_SAN_DNS,
    RESPONSE_MODE_DIRECT_POST, RESPONSE_MODE_DIRECT_POST_JWT,
    RESPONSE_MODE_FRAGMENT, Verifier, VpAuthRequest, VpAuthResponse,
};
pub use wallet::WalletClient;
